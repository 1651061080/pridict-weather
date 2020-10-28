import os

import numpy as np
from PIL import Image
from keras import Model
from keras.models import load_model

from training import DataGenerator
from data_structure import movies as ms
from helpers import paths
from model_structure.model_convLSTM import weighted_mse
import pylab as plt


def main():

    # set parameters
    # movies_tuple = ms.MoviesTuple(path=paths.DATA_FOR_TRAINING, movies_params=ms.get_movies_params())
    # movies_tuple.load_movies()
    # movies_tuple.normalize()
    # predict_from_model(movies_tuple)
    # movies_tuple = ms.MoviesTuple(path=paths.DATA_FOR_VALIDATION, movies_params=ms.get_movies_params())
    # movies_tuple.load_movies()
    # movies_tuple.normalize()
    batch_size=1
    validation_generator = DataGenerator(path=".." + os.sep + paths.DATA_FOR_TESTING, params=ms.get_movies_params(), batch_size=batch_size)

    predict_from_model(validation_generator)


class ImageRange:
    def __init__(self, first, range):
        self.first = first
        self.range = range


def predict_from_model(validation_generator):

    # model_folder = input("Enter name (datetime) of folder containing models located in resources/models (format: 20180910.0817)" + os.linesep) + os.sep
    model_folder = '20181130.0113-final_model' + os.sep

    path_models_datetime = os.path.join(".." + os.sep + paths.MODELS, model_folder)  # resources/models/20181022.0055/
    path_prediction_datetime = os.path.join(".." + os.sep + paths.PREDICTION, model_folder) # resources/prediction/20181022.0055/
    paths_models_model_for_epoch = paths.gel_all_models(path_models_datetime)

    if not paths_models_model_for_epoch:
        print("There is no model in", path_models_datetime)
    else:
        if not os.path.exists(path_prediction_datetime):
            os.makedirs(path_prediction_datetime)
        for path_model in paths_models_model_for_epoch:  # model_path='resources/models/20181022.0055/model.epoch-02-val_loss-0.0185.hdf5'
            folder_model_epoch = os.path.basename(path_model)  # folder_model_epoch='model.epoch-02-val_loss-0.0185.hdf5'
            #  path_prediction_model_epoch='resources/prediction/20181022.0055/model.epoch-02-val_loss-0.0185.hdf5'
            path_prediction_model_epoch = path_prediction_datetime + folder_model_epoch + os.sep
            if not os.path.exists(path_prediction_model_epoch):
                os.makedirs(path_prediction_model_epoch)
                rows=90 # TODO
                loaded_model = load_model(path_model, custom_objects={'weighted_mse': weighted_mse})  # TODO check it
                predict_images(model=loaded_model, validation_generator=validation_generator, save_path=path_prediction_model_epoch)


def predict_images(model, validation_generator, save_path):
    """
    Predict only one image ahead for every sample in range from_image to to_image
    """
    # TODO remove from_image to to_image
    """ INTERMEDIATE LAYER """
    layer_name = 'conv_lst_m2d_7'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict_generator(generator=validation_generator)

    # predicted_images = model.predict_generator(generator=validation_generator)
    predicted_images = intermediate_output

    """ -- INTERMEDIATE LAYER """

    sequence_list = validation_generator.sequence_list
    n_sequences = len(sequence_list)
    for i in range(n_sequences):
        datetime_name = paths.get_name(sequence_list[i])
        # datetime_name = paths.get_datetime_from_name(sequence_list[i])  # TODO int this case datetime is found in path, not only in name
        datetime_path = save_path + datetime_name + os.sep
        if not os.path.exists(datetime_path):
            os.makedirs(datetime_path)

        full_path = datetime_path + datetime_name + '_p_' + '.png'
        predicted_image = predicted_images[i, ::, ::, 0]  # (70,9,64,64,16)

        # TODO for featrue maps -- more dimensions + different definition interval
        """ INTERMEDIATE LAYER PREDICTION """
        predicted_image = predicted_images[8,5, ::, ::, 7]
        plt.imshow(predicted_image, cmap='gray')
        """ -- INTERMEDIATE LAYER PREDICTION """

        final_image = postprocess_image(predicted_image)
        final_image.save(full_path)

    # # plt.imshow(predicted_image[25,8,::,::,0])
    # image_range = ImageRange(first=0, range=movies_tuple.movies_params.samples_ratio.n_samples_loaded)
    # for i_sample in range(image_range.first, image_range.range):
    #     track = movies_tuple.initial_movies.get_input_frames(i_sample)
    #     new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
    #     new = ms.get_last_frame(new_pos)  # new = new_pos[::, -1, ::, ::, ::]  # -1 is for last frame (last predicted frame)
    #     predicted_image = new[0, ::, ::, 0]
    #
    #     prediction_for_image_name = movies_tuple.shifted_movies.names[i_sample, -1]  # 'row0001_col0001_20180722.0710.png'
    #     save_predicted_image(predicted_image, prediction_for_image_name, save_path)


def save_predicted_image(predicted_image, prediction_for_image_name, save_path):

    prediction_for_datetime = paths.get_datetime_from_name(prediction_for_image_name)  # '20180722.0710.png'
    folder = save_path + prediction_for_datetime  # folder is datetime path
    if not os.path.exists(folder):
        os.makedirs(folder)

    prediction_for_row_col = paths.get_row_col_from_name(prediction_for_image_name)
    subfolder = folder + os.sep + prediction_for_row_col  # subfolder is row_col path
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # SAVE PREDICTED IMAGE
    image_path = subfolder + os.sep + prediction_for_row_col + '_' + prediction_for_datetime + '.png'
    final_image = postprocess_image(predicted_image)
    final_image.save(image_path)


def postprocess_image(predicted_image):
    predicted_image -= 0.25
    predicted_image *= 2.0
    predicted_image = np.clip(predicted_image, 0.0, 1.0)  # TODO
    image_0_to_15 = predicted_image * 15
    image_0_to_15_rounded = np.round(image_0_to_15)
    image_0_to_15_uint8 = image_0_to_15_rounded.astype(np.uint8)
    image_0_to_255_uint8 = image_0_to_15_uint8 * 17
    final_image = Image.fromarray(image_0_to_255_uint8)
    return final_image


if __name__ == '__main__':
    main()

