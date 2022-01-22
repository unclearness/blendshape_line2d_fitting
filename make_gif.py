from PIL import Image, ImageDraw
import os
import re

def natural_sort(l):
    def alphanum_key(s):
        return [int(c) if c.isdecimal() else c for c in re.split('([0-9]+)', s) ]
    return sorted(l, key=alphanum_key)

def make_gif(root_dir, prefix, ext):
  images = []

  im_files = natural_sort([x for x in os.listdir(root_dir) if x.startswith(prefix) and x.endswith(ext)])

  for im_file in im_files:
    path = os.path.join(root_dir, im_file)
    im = Image.open(path)
    org_size = im.size
    im = im.resize((int(org_size[0]/2), int(org_size[1]/2)))
    images.append(im)

  images[0].save(prefix + '.gif',
                save_all=True, append_images=images[1:],
                optimize=False, duration=20, loop=0)


if __name__ == "__main__":
  make_gif('./out/', 'combined', 'png')