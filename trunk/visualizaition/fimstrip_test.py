import glob
import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from data_structure import movies as ms
from helpers import paths
from helpers.paths import get_all_subfolders
from helpers.paths import get_name
from training import normalize

datetime_folder = input("Enter name (datetime) of folder containing predictions located in resources/prediction (format: 20180910.0817)" + os.linesep) + os.sep
#datetime_folder = '20181130.0113-final_model' + os.sep

model_folder = input("Enter name of subfolder containing prediction for model for specific epoch located in resources/prediction/datetime (format: model.epoch-03-val_loss-0.0025.hdf5)" + os.linesep) + os.sep
#model_folder = 'model.epoch-10-val_loss-0.0008.hdf5' + os.sep

predictionPath = os.path.join(paths.PREDICTION, datetime_folder, model_folder)


inputPath = '..' + os.sep + paths.DATA_FOR_TESTING

save_path = '..' + os.sep + paths.PREDICTION_STRIPS + datetime_folder

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path + model_folder):
    os.makedirs(save_path + model_folder)

params = ms.get_movies_params()
rows = params.frame_dims.rows
cols = params.frame_dims.cols
small_padding = np.full((rows, 3), 127.0)
large_padding = np.full((rows, 10), 127.0)


inpuFolders = get_all_subfolders(inputPath)
inpuFolders.sort()
predictionFolders = get_all_subfolders("../" + predictionPath)
predictionFolders.sort()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(len(inpuFolders)):
    inputImages = glob.glob(inpuFolders[i] + "*.png")
    inputImages.sort()
    predictionImages = glob.glob(predictionFolders[i] + "*.png")
    # predictionImages.sort()

    strip = np.array(Image.open(inputImages[0]))[:rows:, :cols:]
    for j in range(8):
        inputImage = np.array(Image.open(inputImages[j+1]))[:rows:, :cols:]

        strip = np.concatenate((strip, small_padding), axis=1)
        strip = np.concatenate((strip, inputImage), axis=1)

    # ground truth
    groundtruthImage = np.array(Image.open(inputImages[9]))[:rows:, :cols:]
    # groundtruthImage.fill(200.0)
    strip = np.concatenate((strip, large_padding), axis=1)
    strip = np.concatenate((strip, groundtruthImage), axis=1)

    # prediction
    predictionImage = np.array(Image.open(predictionImages[0]))[:rows:, :cols:]
    # predictionImage = np.array(Image.open(inputImages[8]))[:rows:, :cols:]
    # predictionImage2.fill(200.0)
    strip = np.concatenate((strip, large_padding), axis=1)
    strip = np.concatenate((strip, predictionImage), axis=1)

    # overlay of prediction and ground truth
    rgbOverlay = np.stack([groundtruthImage, predictionImage, predictionImage], axis=2)
    # rgbOverlay = rgb.astype(np.uint8)
    # rgbOverlay = Image.fromarray(rgb_uint8,'RGB')
    # rgbOverlay.show()
    strip = np.concatenate((strip, large_padding), axis=1)
    strip = np.stack([strip, strip, strip], axis=2)
    strip = np.concatenate((strip, rgbOverlay), axis=1)

    strip_uint8 = strip.astype(np.uint8)
    final_strip = Image.fromarray(strip_uint8, 'RGB')

    # unweighted_loss = K.mean(K.square(y_true_ - y_pred_))
    # TODO / 2 because of normalisation from (0,1) to (0.25,0.75)
    groundtruthNormalized = normalize(groundtruthImage.astype(np.float32))
    predictionNormalized = normalize(predictionImage.astype(np.float32))
    loss_value = K.mean(K.square(groundtruthNormalized[1:-1:, 1:-1:] - predictionNormalized[1:-1:, 1:-1:]))

    d = ImageDraw.Draw(final_strip)
    d.text((765, 2), "%0.4f" % np.round(sess.run(loss_value), decimals=4), fill=(70, 140, 50))

    # datetime_name = get_datetime_from_name(inpuFolders[i]) # TODO
    datetime_name = get_name(inpuFolders[i])  # TODO
    final_strip.save(save_path + model_folder + datetime_name + '.png')

print("finished")
