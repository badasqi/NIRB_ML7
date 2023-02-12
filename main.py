import os
import shutil
from itertools import product
import natsort
import pandas as pd
import mahotas as mt
import glob
import csv
import time
import os
import skops.io as sio
import cv2
from PIL import Image, ImageDraw

tic = time.time()
def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)
    wh = (w, h)
    return wh

def concat(dir_out,dir_in,listdir, wh):
    total_width, max_height = wh
    total_width = total_width - total_width % 32 - 32
    max_height = max_height - max_height % 32 - 32
    result = Image.new('RGBA', (total_width, max_height))  # common canvas
    y_offset = 0
    x_offset = 0
    for image in listdir:
        img = Image.open(os.path.join(dir_in, image))
        if x_offset < total_width:
            result.paste(img, (x_offset, y_offset))
            x_offset += 32
        else:
            y_offset += 32
            x_offset = 0
        img.close()
    result.save(os.path.join(dir_out, 'result.png'))
    result.close()
    print("[STATUS] Image predict saved : test/output/result.png")

# function to extract haralick textures from an image
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

if __name__ == '__main__':
    wh = (640, 424)
    dir_in = r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\test_image'
    os.mkdir(r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\cropp_test_image')
    dir_out = r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\cropp_test_image'
    for filename in os.listdir(dir_in):
        wh = tile(filename, dir_in, dir_out, 32)
    # load the training dataset
    train_path = r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\cropp_test_image'  # Enter the directory where all the images are stored
    train_names = os.listdir(train_path)

    # empty list to hold feature vectors and train labels
    train_features = []
    train_labels = []
    # loop over the training dataset
    print("[STATUS] Started extracting haralick textures..")
    cur_path = os.path.join(train_path, '*g')
    cur_label = train_names
    i = 0
    with open('test_fragments_img.csv', 'a+', newline='') as obj:
        writer = csv.writer(obj)
        if i == 0:
            writer.writerow(
                ['Haralick1', 'Haralick2', 'Haralick3', 'Haralick4', 'Haralick5', 'Haralick6', 'Haralick7', 'Haralick8',
                 'Haralick9',
                 'Haralick10', 'Haralick11', 'Haralick12', 'Haralick13'])
        for file in glob.glob(cur_path):
            # print("Processing Image - {} in {}".format(i, cur_label[i]))
            # read the training image
            image = cv2.imread(file)

            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # extract haralick texture from image
            features = extract_features(gray)
            # print(features)

            # append the feature vector and label
            train_features.append(features)
            train_labels.append(cur_label[i])

            writer.writerow(features)

            # show loop update
            i += 1

    test = pd.read_csv('test_fragments_img.csv', sep=',', header=0)

    # LOAD MODELs
    with open('models/Haralick_LGBM_model.pkl', 'rb') as f:
        model = sio.load(f, trusted=True)
    model_prediction = list(model.predict(test))
    list = []
    m = 0
    for i in model_prediction:
        m += 1
        if i == 1:
            list.append(m)
    i = 0
    TINT_COLOR = (148, 0, 211)
    TRANSPARENCY = .80  # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)
    os.mkdir(r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\predict')
    listdir = natsort.natsorted(os.listdir(r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\cropp_test_image'), reverse=False)
    print("[STATUS] Started predict textures..")
    for images in listdir:
        with Image.open(os.path.join('C:/Users/Ilya/PycharmProjects/NIRB_ML7/test/cropp_test_image/', images)) as im:
            height, width = im.size
            im = im.convert("RGBA")
            i += 1
            if i in list:
                overlay = Image.new('RGBA', im.size, TINT_COLOR + (0,))
                draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
                draw.rectangle(((0, 0), (32, 32)), fill=TINT_COLOR + (OPACITY,))

                # Alpha composite these two images together to obtain the desired result.
                im = Image.alpha_composite(im, overlay)
                im = im.convert("RGB")  # Remove alpha for saving in jpg format.
            im.save(r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\predict\predict_image_' + str(i) + '_.png')
    dir_in = r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\predict'
    dir_path1 = 'C:/Users/Ilya/PycharmProjects/NIRB_ML7/test/predict'
    dir_path2 = 'C:/Users/Ilya/PycharmProjects/NIRB_ML7/test/cropp_test_image'
    dir_out = r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test\output_image'
    files = os.listdir(dir_in)
    files = natsort.natsorted(files, reverse=False)
    concat(dir_out, dir_in, files, wh)
    try:
        shutil.rmtree(dir_path1)
    except OSError as e:
        print("Ошибка: %s : %s" %(dir_path1, e.strerror))
    dir_path2 = 'C:/Users/Ilya/PycharmProjects/NIRB_ML7/test/cropp_test_image'
    try:
        shutil.rmtree(dir_path2)
    except OSError as e:
        print("Ошибка: %s : %s" % (dir_path2, e.strerror))
    os.remove(r'C:\Users\Ilya\PycharmProjects\NIRB_ML7\test_fragments_img.csv')