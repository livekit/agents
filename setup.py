#
#  Copyright © 2025 Agora
#  This file is part of TEN Framework, an open source project.
#  Licensed under the Apache License, Version 2.0, with certain conditions.
#  Refer to the "LICENSE" file in the root directory for more information.
#
from setuptools import setup
import os, shutil, platform
from setuptools.command.install import install

class custom_install_command(install):
    def run(self):
        install.run(self)
        target_dir = os.path.join(self.install_lib, "ten_vad_library")
        os.makedirs(target_dir, exist_ok=True)
        
        if platform.system() == "Linux" and platform.machine() == "x86_64":
            shutil.copy("lib/Linux/x64/libten_vad.so", target_dir)
            print(f"Linux x64 library installed to: {target_dir}")
        elif platform.system() == "Darwin":
            shutil.copy("lib/macOS/ten_vad.framework/Versions/A/ten_vad", 
                       os.path.join(target_dir, "libten_vad"))
            print(f"macOS library installed to: {target_dir}")
        elif platform.system().upper() == 'WINDOWS':
            if platform.machine().upper() in ['X64', 'X86_64', 'AMD64']:
                shutil.copy("lib/Windows/x64/ten_vad.dll", 
                       os.path.join(target_dir, "ten_vad.dll"))
                print(f"Windows x64 library installed to: {target_dir}")
            else:
                shutil.copy("lib/Windows/x86/ten_vad.dll", 
                       os.path.join(target_dir, "ten_vad.dll"))
                print(f"Windows x86 library installed to: {target_dir}")
        else:
            raise NotImplementedError(f"Unsupported platform: {platform.system()} {platform.machine()}")

root_dir = os.path.dirname(os.path.abspath(__file__))
shutil.copy(f"{root_dir}/include/ten_vad.py", f"{root_dir}/ten_vad.py")
setup(
    name="ten_vad",
    version="1.0",
    py_modules=["ten_vad"],
    cmdclass={
        "install": custom_install_command,
    },
)
os.remove(f"{root_dir}/ten_vad.py")