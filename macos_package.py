import os
import sys
cmd = "python3.9 setup.py py2app --packages=PyQt5"
os.system(cmd)
cmd = "cp /Users/yipengzhang/miniconda3/envs/hfo/lib/liblzma.5.dylib dist/main.app/Contents/Frameworks" 
os.system(cmd)
cmd = "mv ./dist/main.app/Contents/Resources/ckpt ./dist/main.app/Contents/Resources/lib/python3.9/"
os.system(cmd)
cmd = "./dist/main.app/Contents/MacOS/main"
os.system(cmd)