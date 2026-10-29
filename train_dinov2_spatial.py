"""Entrada al pipeline integrado. Usa los mismos argumentos de deepz_pipeline.py.

python train_dinov2_spatial.py --modelo dinov2_convnext_tiny --dataset ../dataset
python train_dinov2_spatial.py --modelo dinov2_swin_tiny --dataset ../dataset
"""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
    from deepz_pipeline import main
    if not any(arg == "--modelo" or arg.startswith("--modelo=") for arg in sys.argv[1:]):
        sys.argv.extend(["--modelo", "dinov2_convnext_tiny"])
    main()
