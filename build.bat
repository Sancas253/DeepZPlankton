@echo off
REM Genera el ejecutable de DeepZPlankton en dist\DeepZPlanktonV3\
REM Requiere Python 3.10 en el PATH.

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pyinstaller --noconfirm DeepZPlankton_V2.spec

echo.
echo Listo: dist\DeepZPlanktonV3\DeepZPlanktonV3.exe
echo Copia la carpeta del modelo (con los .onnx) en dist\DeepZPlanktonV3\model\
pause
