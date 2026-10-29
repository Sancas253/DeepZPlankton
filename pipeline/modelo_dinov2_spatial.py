"""Arquitectura compartida por entrenamiento y exportación ONNX."""
import torch
from torch import nn
import timm
from timm.optim import param_groups_layer_decay, param_groups_weight_decay
from transformers import Dinov2Config, Dinov2Model


class DINOv2Spatial(nn.Module):
    def __init__(self, num_classes, backbone="convnext_tiny", image_size=224,
                 pretrained=True, drop_path_rate=0.1, transfer=True, classifier_weights=None):
        super().__init__()
        if image_size < 112 or image_size % 112:
            raise ValueError("IMAGE_SIZE debe ser múltiplo de 112 (patch 14 y tres reducciones x2)")
        if pretrained:
            self.dinov2 = Dinov2Model.from_pretrained(
                "facebook/dinov2-base", attn_implementation="eager")
        else:
            # facebook/dinov2-base conserva posiciones 37x37 aunque la entrada sea 224.
            config = Dinov2Config(image_size=518)
            config._attn_implementation = "eager"
            self.dinov2 = Dinov2Model(config)
        self.dinov2.requires_grad_(False)
        self.dinov2.eval()
        self.grid = image_size // self.dinov2.config.patch_size
        channels = self.dinov2.config.hidden_size
        self._adapter_only = False
        if not transfer:
            # Compatibilidad con checkpoints de la primera variante, sin preentrenamiento.
            if backbone == "convnext_tiny":
                self.classifier = timm.create_model(
                    "convnext_tiny", pretrained=False, in_chans=channels,
                    patch_size=1, num_classes=num_classes)
            elif backbone == "swin_tiny":
                self.classifier = timm.create_model(
                    "swin_tiny_patch4_window7_224", pretrained=False,
                    in_chans=channels, img_size=self.grid, patch_size=1,
                    window_size=4, num_classes=num_classes)
            else:
                raise ValueError(f"Backbone desconocido: {backbone}")
            self.adapter = nn.Identity()
            return
        # Se cargan los pesos RGB ANTES de sustituir la entrada por el adaptador.
        if backbone == "convnext_tiny":
            weights = classifier_weights or "convnext_tiny.fb_in22k_ft_in1k"
            if weights.split('.')[0] != "convnext_tiny":
                raise ValueError("PESOS debe ser una variante de convnext_tiny")
            self.classifier = timm.create_model(
                weights, pretrained=pretrained,
                num_classes=num_classes, drop_path_rate=drop_path_rate)
            width = self.classifier.stem[0].out_channels
            self.classifier.stem[0] = nn.Identity()
        elif backbone == "swin_tiny":
            weights = classifier_weights or "swin_tiny_patch4_window7_224.ms_in1k"
            if weights.split('.')[0] != "swin_tiny_patch4_window7_224":
                raise ValueError("PESOS debe ser una variante de swin_tiny_patch4_window7_224")
            self.classifier = timm.create_model(
                weights, pretrained=pretrained,
                num_classes=num_classes, drop_path_rate=drop_path_rate)
            self.classifier.set_input_size(img_size=(self.grid, self.grid),
                                           patch_size=(1, 1), window_size=(4, 4))
            width = self.classifier.patch_embed.proj.out_channels
            self.classifier.patch_embed.proj = nn.Identity()
        else:
            raise ValueError(f"Backbone desconocido: {backbone}")
        self.adapter = nn.Conv2d(channels, width, kernel_size=1)

    def set_adapter_only(self, enabled):
        self._adapter_only = enabled
        self.classifier.requires_grad_(not enabled)
        self.classifier.get_classifier().requires_grad_(True)

    def make_optimizer(self, lr, new_lr, weight_decay, layer_decay):
        groups = param_groups_layer_decay(
            self.classifier, weight_decay=weight_decay, layer_decay=layer_decay,
            no_weight_decay_list=self.classifier.no_weight_decay()
            if hasattr(self.classifier, "no_weight_decay") else ())
        head_ids = {id(p) for p in self.classifier.get_classifier().parameters()}
        split_groups = []
        for group in groups:
            old = [p for p in group['params'] if id(p) not in head_ids]
            new = [p for p in group['params'] if id(p) in head_ids]
            if old:
                split_groups.append(dict(group, params=old))
            if new:
                split_groups.append(dict(group, params=new, lr_scale=new_lr / lr))
        for group in param_groups_weight_decay(self.adapter, weight_decay=weight_decay):
            split_groups.append(dict(group, lr_scale=new_lr / lr))
        # CosineLRScheduler aplica lr_scale también durante el warmup.
        return torch.optim.AdamW(split_groups, lr=lr)

    def train(self, mode=True):
        super().train(mode)
        self.dinov2.eval()
        if mode and self._adapter_only:
            self.classifier.eval()
            self.classifier.get_classifier().train()
        return self

    def forward(self, images):
        with torch.no_grad():
            tokens = self.dinov2(pixel_values=images, return_dict=False)[0][:, 1:]
        patches = tokens.transpose(1, 2).reshape(
            images.shape[0], self.dinov2.config.hidden_size, self.grid, self.grid)
        return self.classifier(self.adapter(patches))


class DINOv2ConvNeXtPlus(nn.Module):
    """DINOv2 congelado -> etapas 2-3 de ConvNeXt-Tiny preentrenado, + CLS en la cabeza.

    Los patches de DINOv2 (stride 14) entran donde ConvNeXt ya trabaja a stride 16
    con 384 canales (etapa 2), en vez de reemplazar el stem de stride 4: así se
    conservan los bloques que corresponden a esa escala y el adaptador comprime
    768->384 en lugar de 768->96. Se mezclan las últimas N_LAYERS capas de DINOv2
    (pesos aprendidos, como en la evaluación lineal de DINOv2) y el token CLS se
    concatena al pooling de ConvNeXt antes del clasificador.
    """
    N_LAYERS = 4

    def __init__(self, num_classes, image_size=280, pretrained=True, drop_path_rate=0.1,
                 classifier_weights=None, head_dropout=0.2):
        super().__init__()
        if image_size < 28 or image_size % 28:
            raise ValueError("IMAGE_SIZE debe ser múltiplo de 28 (patch 14 y una reducción x2)")
        if pretrained:
            self.dinov2 = Dinov2Model.from_pretrained(
                "facebook/dinov2-base", attn_implementation="eager")
        else:
            config = Dinov2Config(image_size=518)
            config._attn_implementation = "eager"
            self.dinov2 = Dinov2Model(config)
        self.dinov2.requires_grad_(False)
        self.dinov2.eval()
        self.grid = image_size // self.dinov2.config.patch_size
        channels = self.dinov2.config.hidden_size
        self.n_layers = min(self.N_LAYERS, self.dinov2.config.num_hidden_layers)
        self._adapter_only = False

        weights = classifier_weights or "convnext_tiny.fb_in22k_ft_in1k"
        if weights.split('.')[0] != "convnext_tiny":
            raise ValueError("PESOS debe ser una variante de convnext_tiny")
        # num_classes=0: la cabeza de timm queda en pooling + LayerNorm (pre-logits).
        self.classifier = timm.create_model(weights, pretrained=pretrained, num_classes=0,
                                            drop_path_rate=drop_path_rate)
        width = self.classifier.stages[2].downsample[1].out_channels
        self.classifier.stem = nn.Identity()
        self.classifier.stages[0] = nn.Identity()
        self.classifier.stages[1] = nn.Identity()
        self.classifier.stages[2].downsample = nn.Identity()

        self.layer_mix = nn.Parameter(torch.zeros(self.n_layers))
        self.adapter = nn.Sequential(timm.layers.LayerNorm2d(channels),
                                     nn.Conv2d(channels, width, kernel_size=1))
        self.cls_norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(head_dropout)
        self.fc = nn.Linear(self.classifier.num_features + channels, num_classes)

    def new_modules(self):
        return nn.ModuleList([self.adapter, self.cls_norm, self.fc])

    def set_adapter_only(self, enabled):
        self._adapter_only = enabled
        self.classifier.requires_grad_(not enabled)

    def make_optimizer(self, lr, new_lr, weight_decay, layer_decay):
        groups = param_groups_layer_decay(
            self.classifier, weight_decay=weight_decay, layer_decay=layer_decay,
            no_weight_decay_list=self.classifier.no_weight_decay()
            if hasattr(self.classifier, "no_weight_decay") else ())
        new_groups = param_groups_weight_decay(self.new_modules(), weight_decay=weight_decay)
        new_groups.append({"params": [self.layer_mix], "weight_decay": 0.0})
        for group in new_groups:
            groups.append(dict(group, lr_scale=new_lr / lr))
        return torch.optim.AdamW(groups, lr=lr)

    def train(self, mode=True):
        super().train(mode)
        self.dinov2.eval()
        if mode and self._adapter_only:
            self.classifier.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            hidden = self.dinov2(pixel_values=images, output_hidden_states=True,
                                 return_dict=True).hidden_states[-self.n_layers:]
            layers = torch.stack([self.dinov2.layernorm(h) for h in hidden])
        mix = torch.softmax(self.layer_mix, dim=0).to(layers.dtype)
        tokens = (mix[:, None, None, None] * layers).sum(0)
        patches = tokens[:, 1:].transpose(1, 2).reshape(
            images.shape[0], tokens.shape[-1], self.grid, self.grid)
        x = self.classifier.stages[3](self.classifier.stages[2](self.adapter(patches)))
        x = self.classifier.forward_head(x)
        return self.fc(self.drop(torch.cat([x, self.cls_norm(tokens[:, 0])], dim=1)))


class DINOv2FTConvNeXt(nn.Module):
    """DINOv2 ViT-B/14 con fine-tuning (como dinov2_ft) -> etapas 2-3 de ConvNeXt-Tiny.

    Misma cabeza que DINOv2ConvNeXtPlus (últimas 4 capas mezcladas + CLS), pero
    el ViT es el de timm de dinov2_ft y se entrena: las features se adaptan al
    plancton en vez de quedar fijas. Durante set_adapter_only(True) ambos
    backbones quedan congelados para que el adaptador recién inicializado no
    mande gradientes ruidosos al ViT.
    """
    N_LAYERS = 4

    def __init__(self, num_classes, image_size=224, pretrained=True, drop_path_rate=0.1,
                 vit_drop_path_rate=0.1, vit_weights=None, classifier_weights=None,
                 head_dropout=0.2):
        super().__init__()
        if image_size < 28 or image_size % 28:
            raise ValueError("IMAGE_SIZE debe ser múltiplo de 28 (patch 14 y una reducción x2)")
        vit_weights = vit_weights or "vit_base_patch14_dinov2.lvd142m"
        self.vit = timm.create_model(vit_weights, pretrained=pretrained, num_classes=0,
                                     img_size=image_size, drop_path_rate=vit_drop_path_rate)
        channels = self.vit.embed_dim
        self.n_layers = min(self.N_LAYERS, len(self.vit.blocks))
        self._adapter_only = False

        weights = classifier_weights or "convnext_tiny.fb_in22k_ft_in1k"
        if weights.split('.')[0] != "convnext_tiny":
            raise ValueError("PESOS debe ser una variante de convnext_tiny")
        self.classifier = timm.create_model(weights, pretrained=pretrained, num_classes=0,
                                            drop_path_rate=drop_path_rate)
        width = self.classifier.stages[2].downsample[1].out_channels
        self.classifier.stem = nn.Identity()
        self.classifier.stages[0] = nn.Identity()
        self.classifier.stages[1] = nn.Identity()
        self.classifier.stages[2].downsample = nn.Identity()

        self.layer_mix = nn.Parameter(torch.zeros(self.n_layers))
        self.adapter = nn.Sequential(timm.layers.LayerNorm2d(channels),
                                     nn.Conv2d(channels, width, kernel_size=1))
        self.cls_norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(head_dropout)
        self.fc = nn.Linear(self.classifier.num_features + channels, num_classes)

    def new_modules(self):
        return nn.ModuleList([self.adapter, self.cls_norm, self.fc])

    def set_adapter_only(self, enabled):
        self._adapter_only = enabled
        self.vit.requires_grad_(not enabled)
        self.classifier.requires_grad_(not enabled)

    def make_optimizer(self, lr, new_lr, weight_decay, layer_decay, vit_lr, vit_layer_decay):
        groups = []
        for module, base, decay in ((self.vit, vit_lr, vit_layer_decay),
                                    (self.classifier, lr, layer_decay)):
            for group in param_groups_layer_decay(
                    module, weight_decay=weight_decay, layer_decay=decay,
                    no_weight_decay_list=module.no_weight_decay()
                    if hasattr(module, "no_weight_decay") else ()):
                groups.append(dict(group, lr_scale=group.get("lr_scale", 1.0) * base / lr))
        new_groups = param_groups_weight_decay(self.new_modules(), weight_decay=weight_decay)
        new_groups.append({"params": [self.layer_mix], "weight_decay": 0.0})
        for group in new_groups:
            groups.append(dict(group, lr_scale=new_lr / lr))
        return torch.optim.AdamW(groups, lr=lr)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._adapter_only:
            self.vit.eval()
            self.classifier.eval()
        return self

    def forward(self, images):
        feats = self.vit.forward_intermediates(
            images, indices=self.n_layers, norm=True, output_fmt="NCHW",
            return_prefix_tokens=True, intermediates_only=True)
        mix = torch.softmax(self.layer_mix, dim=0)
        patches = sum(w * f for w, (f, _) in zip(mix, feats))
        cls = sum(w * p[:, 0] for w, (_, p) in zip(mix, feats))
        x = self.classifier.stages[3](self.classifier.stages[2](self.adapter(patches)))
        x = self.classifier.forward_head(x)
        return self.fc(self.drop(torch.cat([x, self.cls_norm(cls)], dim=1)))


class ConvNeXtDINOv2FusionPlus(nn.Module):
    """ConvNeXt-Tiny completo y DINOv2 ViT-B/14 en paralelo, ambos con fine-tuning.

    Cada rama mira la imagen original a la resolución con la que funcionó sola:
    ConvNeXt a image_size (288, como convnext_tiny_plus) y el ViT a vit_size
    (224, como dinov2_ft; se reescala adentro del modelo, así el ONNX recibe
    una sola entrada). Se concatenan el pooling de ConvNeXt y el CLS del ViT.
    Durante set_adapter_only(True) solo se entrena la cabeza.
    """

    def __init__(self, num_classes, image_size=288, vit_size=224, pretrained=True,
                 drop_path_rate=0.1, vit_drop_path_rate=0.1, vit_weights=None,
                 classifier_weights=None, head_dropout=0.2):
        super().__init__()
        if vit_size % 14:
            raise ValueError("VIT_SIZE debe ser múltiplo de 14")
        self.vit_size = vit_size
        self.vit = timm.create_model(vit_weights or "vit_base_patch14_dinov2.lvd142m",
                                     pretrained=pretrained, num_classes=0, img_size=vit_size,
                                     drop_path_rate=vit_drop_path_rate)
        weights = classifier_weights or "convnext_tiny.fb_in22k_ft_in1k"
        if weights.split('.')[0] != "convnext_tiny":
            raise ValueError("PESOS debe ser una variante de convnext_tiny")
        self.classifier = timm.create_model(weights, pretrained=pretrained, num_classes=0,
                                            drop_path_rate=drop_path_rate)
        self._adapter_only = False
        self.drop = nn.Dropout(head_dropout)
        self.fc = nn.Linear(self.classifier.num_features + self.vit.num_features, num_classes)

    def set_adapter_only(self, enabled):
        self._adapter_only = enabled
        self.vit.requires_grad_(not enabled)
        self.classifier.requires_grad_(not enabled)

    def make_optimizer(self, lr, new_lr, weight_decay, layer_decay, vit_lr, vit_layer_decay):
        groups = []
        for module, base, decay in ((self.vit, vit_lr, vit_layer_decay),
                                    (self.classifier, lr, layer_decay)):
            for group in param_groups_layer_decay(
                    module, weight_decay=weight_decay, layer_decay=decay,
                    no_weight_decay_list=module.no_weight_decay()
                    if hasattr(module, "no_weight_decay") else ()):
                groups.append(dict(group, lr_scale=group.get("lr_scale", 1.0) * base / lr))
        for group in param_groups_weight_decay(self.fc, weight_decay=weight_decay):
            groups.append(dict(group, lr_scale=new_lr / lr))
        return torch.optim.AdamW(groups, lr=lr)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._adapter_only:
            self.vit.eval()
            self.classifier.eval()
        return self

    def forward(self, images):
        vit_in = images
        if images.shape[-1] != self.vit_size or images.shape[-2] != self.vit_size:
            vit_in = nn.functional.interpolate(images, size=(self.vit_size, self.vit_size),
                                               mode="bilinear", align_corners=False)
        cls = self.vit(vit_in)
        cnx = self.classifier(images)
        return self.fc(self.drop(torch.cat([cnx, cls], dim=1)))
