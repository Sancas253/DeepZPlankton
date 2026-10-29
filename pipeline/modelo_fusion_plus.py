"""Arquitectura de convnext_dinov2_fusion_plus, compartida por entrenamiento y exportación ONNX."""
import torch
from torch import nn
import timm
from timm.optim import param_groups_layer_decay, param_groups_weight_decay
from transformers import Dinov2Config, Dinov2Model


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
