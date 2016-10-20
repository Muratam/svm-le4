from PIL import Image
import sys
from random import randint

if __name__ == "__main__":
    assert len(sys.argv) >= 3
    img = Image.open(sys.argv[1])
    img.thumbnail((100, 100))
    img = img.convert("L")  # 1 : 01 | L : grayscale
    w, h = img.size
    for i in range(int(sys.argv[2])):
        x = randint(0, w - 1)
        y = randint(0, h - 1)
        which = img.getpixel((x, y))
        which = 1 if which < 128 else -1
        res = "{} {} {}".format(x, y, which)
        print(res)
