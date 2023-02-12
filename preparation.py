import random
import numpy as np
import os
from PIL import Image, ImageDraw
from natsort import natsort

i = 0
listdir = natsort.natsorted(os.listdir(r'...'), reverse=False) #enter image path

for images in listdir:
    im = Image.open(os.path.join(r'D:\image\train_img\new_new_img',images))
    height, width = im.size
    draw = ImageDraw.Draw(im)
    ellipse1_h, ellipse1_w = [64, 108], [128, 177]
    draw.ellipse((ellipse1_h[0], ellipse1_w[0], ellipse1_h[1], ellipse1_w[1]), fill='white',
                 outline='white')
    rectangle_h, rectangle_w = [128, 178], [192, 248]
    draw.rectangle((rectangle_h[0], rectangle_w[0], rectangle_h[1], rectangle_w[1]), fill='white',
                   outline='white')
    ellipse2_h, ellipse2_w = [160, 204], [288, 342]
    draw.ellipse((ellipse2_h[0], ellipse2_w[0], ellipse2_h[1], ellipse2_w[1]), fill='white',
                 outline='white')
    rectangle1_h, rectangle1_w = [384, 448], [256, 320]
    draw.rectangle((rectangle1_h[0], rectangle1_w[0], rectangle1_h[1], rectangle1_w[1]), fill='white',
                   outline='white')
    rectangle2_h, rectangle2_w = [384, 448], [480, 544]
    draw.rectangle((rectangle2_h[0], rectangle2_w[0], rectangle2_h[1], rectangle2_w[1]), fill='white',
                   outline='white')
    i += 1
    print(i)
    im.save(r'D:\image\train_img\new_new_incomplete\incomplete_image_'+str(i)+'.png', quality=95)
    converted = np.zeros((width, height, 3), dtype="uint8")
    im = Image.fromarray(converted)
    draw = ImageDraw.Draw(im)
    draw.ellipse((ellipse1_h[0], ellipse1_w[0], ellipse1_h[1], ellipse1_w[1]), fill=(255, 255, 255),
                 outline=(255, 255, 255))
    draw.rectangle((rectangle_h[0], rectangle_w[0], rectangle_h[1], rectangle_w[1]), fill=(255, 255, 255),
                   outline=(255, 255, 255))
    draw.ellipse((ellipse2_h[0], ellipse2_w[0], ellipse2_h[1], ellipse2_w[1]), fill=(255, 255, 255),
                 outline=(255, 255, 255))
    draw.rectangle((rectangle1_h[0], rectangle1_w[0], rectangle1_h[1], rectangle1_w[1]), fill=(255, 255, 255),
                   outline=(255, 255, 255))
    draw.rectangle((rectangle2_h[0], rectangle2_w[0], rectangle2_h[1], rectangle2_w[1]), fill=(255, 255, 255),
                   outline=(255, 255, 255))
    im.save(r'D:\image\train_img\new_new_mask\mask_image_' + str(i) + '.png', quality=95)

