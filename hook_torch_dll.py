import os
import sys

# Add torch/lib to PATH so Windows can find c10.dll and its dependencies
if hasattr(sys, "_MEIPASS"):
    torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
