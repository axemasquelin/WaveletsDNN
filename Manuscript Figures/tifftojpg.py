import os
from PIL import Image

cwd = os.getcwd()
for root, dirs, files in os.walk(cwd, topdown=False):
    for name in files:
        im = Image.open(name)
        im.save(os.path.splitext(name)[0] + '.png')
