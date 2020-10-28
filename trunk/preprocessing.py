import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_structure import movies as ms
from helpers import paths

sequence_list = []
y_list = []
path = "/media/vladka/30343CEF343CB9A4/Users/vladi/Desktop/Metoeradar1/09_data_augmentation/"

folders_paths = paths.get_all_subfolders(path)
folders_paths.sort()
n_folders = len(folders_paths)
for i_folder in range(n_folders):
    """ row0081_col0041, row0041_col0361, ... sorted """
    subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
    subfolders_paths.sort()
    n_subfolders = len(subfolders_paths)
    sequence_list.extend(subfolders_paths)


#
#
# # datetime_folder = input("Enter name (datetime) of folder containing predictions located in resources/prediction (format: 20180910.0817)" + os.linesep) + os.sep
# # # datetime_folder = '20181025.2126' + os.sep
# #
# # model_folder = input("Enter name of subfolder containing prediction for model for specific epoch located in resources/prediction/datetime (format: model.epoch-03-val_loss-0.0025.hdf5)" + os.linesep) + os.sep
# # # model_folder = 'model.epoch-02-val_loss-0.0008.hdf5' + os.sep
#
# # prediction_folder = os.path.join(paths.PREDICTION, datetime_folder, model_folder)
#
# loss_weights = ms.Movies(path=paths.MASKS, movies_params=ms.get_movies_params())
# loss_weights.load_movies_loss_weights()
# loss_weights.normalize()
#
#
# y_true=mpimg.imread("/home/vladka/Desktop/DP/Projects/xhezelov_nowcasting/trunk/resources/radar_data/validation_data/validation_data_90x90/20180725.1440/row0001_col0181/row0001_col0181_20180725.1430.png")
# y_pred=mpimg.imread("/home/vladka/Desktop/DP/Projects/xhezelov_nowcasting/trunk/resources/radar_data/validation_data/validation_data_90x90/20180725.1440/row0001_col0181/row0001_col0181_20180725.1440.png")
#
# def to_binary(img, trashold):
#     return trashold < img
#
# binary_true = to_binary(y_true, 0.01)
# imgplot1 = plt.imshow(binary_true)
#
# plt.figure()
#
# binary_pred = to_binary(y_pred, 0.01)
# imgplot2 = plt.imshow(binary_pred)
#
# intersection = binary_true & binary_pred
# diff = binary_true ^ binary_pred
# complement = np.invert(intersection + diff)
# mask = 1*diff + 0.3*intersection + 0.1*complement
# plt.imshow(mask, cmap='gray')
#
# mse = mask * tf.reduce_mean(tf.square(y_true - y_pred))
#
#
# k = 8
#
#
