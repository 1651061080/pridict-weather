from helpers import paths
from data_structure import movies as ms
import pylab as plt
import numpy as np
from PIL import Image
import os


# datetime_folder = input("Enter name (datetime) of folder containing predictions located in resources/prediction (format: 20180910.0817)" + os.linesep) + os.sep
datetime_folder = '20181106.1258' + os.sep

# model_folder = input("Enter name of subfolder containing prediction for model for specific epoch located in resources/prediction/datetime (format: model.epoch-03-val_loss-0.0025.hdf5)" + os.linesep) + os.sep
model_folder = 'model.epoch-29-val_loss-0.0177.hdf5' + os.sep

prediction_folder = os.path.join(paths.PREDICTION, datetime_folder, model_folder)

validation_movies_tuple = ms.MoviesTuple(path=".." + os.sep + paths.DATA_FOR_VALIDATION, movies_params=ms.get_movies_params())
validation_movies_tuple.load_movies()
# training_movies_tuple.normalize()

prediction_movies_nn = ms.Movies(path=".." + os.sep + prediction_folder, movies_params=ms.get_movies_params())
prediction_movies_nn.load_movies()
# prediction_movies_nn.normalize()

rows = validation_movies_tuple.movies_params.frame_dims.cols
small_padding = np.full((rows, 3), 255.0)
large_padding = np.full((rows, 10), 255.0)


path = ".." + os.sep + paths.PREDICTION_STRIPS
datetime_path = path + datetime_folder
if not os.path.exists(datetime_path):
    os.makedirs(datetime_path)

model_path = datetime_path + model_folder
if not os.path.exists(model_path):
    os.makedirs(model_path)

for i_sample in range(0, validation_movies_tuple.movies_params.samples_ratio.n_samples_loaded):

    strip = validation_movies_tuple.initial_movies.get_frame(i_sample, 0)

    # input images
    for i_frame in range(1, validation_movies_tuple.movies_params.frames_ratio.n_frames_in):

        frame = validation_movies_tuple.initial_movies.get_frame(i_sample, i_frame)
        strip = np.concatenate((strip, small_padding), axis=1)
        strip = np.concatenate((strip, frame), axis=1)
        # image_0_to_15_uint8 = image_0_to_15_rounded.astype(np.uint8)

    # Ground Truth output
    frame = validation_movies_tuple.shifted_movies.get_frame(i_sample, validation_movies_tuple.movies_params.frames_ratio.n_frames_in - 1)
    strip = np.concatenate((strip, large_padding), axis=1)
    strip = np.concatenate((strip, frame), axis=1)

    # my prediction
    frame = prediction_movies_nn.get_frame(i_sample, 0)
    strip = np.concatenate((strip, large_padding), axis=1)
    strip = np.concatenate((strip, frame), axis=1)

    datetime_prediction = paths.get_datetime_from_name(prediction_movies_nn.get_name(i_sample, 0))
    row_col_prediction = paths.get_row_col_from_name(prediction_movies_nn.get_name(i_sample, 0))
    save_path = model_path + datetime_prediction + "_" + row_col_prediction + ".png"

    strip_uint8 = strip.astype(np.uint8)
    final_strip = Image.fromarray(strip_uint8)
    final_strip.save(save_path)

