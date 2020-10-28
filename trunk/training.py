import glob
import os

import keras
import numpy as np
from PIL import Image
from keras.utils import Sequence

from data_structure import movies as ms
from data_structure.movies import to_binary
from model_structure.helpers import paths
from model_structure.helpers import redirection
from model_structure import model_convLSTM
from keras.models import load_model
from model_structure.model_convLSTM import weighted_mse
from keras import backend as K


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def main():

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))

    paths_MODEL_FOLDER = paths.create_folder_with_timestamp(paths.MODELS)  # 'resources/models/20181023.2304/'
    paths_MODEL = os.path.join(paths_MODEL_FOLDER, "ConvLSTM_model.h5")  # 'resources/models/20181023.2304/ConvLSTM_model.h5'

    paths_LOGS_FOLDER = paths.create_folder_with_timestamp(paths.LOGS)  # 'resources/logs_tensorboard/20181023.2304/'

    """ redirect the stdout to file saved in model folder """
    redir = redirection.StdoutRedirection(paths_MODEL_FOLDER)
    redir.redirect_to_file()

    # training_movies_tuple = ms.MoviesTuple(path=paths.DATA_FOR_TRAINING, movies_params=ms.get_movies_params())
    # training_movies_tuple.load_movies_with_weights()
    # training_movies_tuple.normalize()
    # TODO
    # validation_movies_tuple = ms.MoviesTuple(path=paths.DATA_FOR_VALIDATION, movies_params=ms.get_movies_params())
    # validation_movies_tuple.load_movies_with_weights()
    # validation_movies_tuple.normalize()

    conv_lstm_model = model_convLSTM.get_model()

    # TRAIN THE MODEL

    batch_size = 4
    epochs = 1000
    """ period: how often (after how many epochs) save model """
    period = 10
    print("---------------------------------")
    print("Batch size: ", batch_size)
    print("Epochs: ", epochs)
    print("---------------------------------")
    # print("Train on ", training_movies_tuple.movies_params.samples_ratio.n_samples_loaded)
    # print("Validate on ", validation_movies_tuple.movies_params.samples_ratio.n_samples_loaded)
    print("---------------------------------")

    """ callback for training process"""
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=paths_LOGS_FOLDER, histogram_freq=0, batch_size=batch_size, write_graph=True,
    #                                                    write_grads=True, write_images=True, update_freq='batch')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=paths_LOGS_FOLDER, histogram_freq=0, batch_size=batch_size, write_graph=True,
                                                       write_grads=True, write_images=True, update_freq='epoch')

    """ callback for saving model after 'period' number of epochs"""
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=paths_MODEL_FOLDER + "model.epoch-{epoch:02d}-val_loss-{val_loss:.4f}.hdf5",
                                                                monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=period)
    print_tensorboard_log_info()

    redir.redirect_to_stdout()

    # loss_weights = ms.Movies(path=paths.MASKS, movies_params=ms.get_movies_params())
    # loss_weights.load_movies_loss_weights()
    # loss_weights.normalize()

    training_generator = DataGenerator(path=paths.DATA_FOR_TRAINING, params=ms.get_movies_params(), batch_size=batch_size)
    validation_generator = DataGenerator(path=paths.DATA_FOR_VALIDATION, params=ms.get_movies_params(), batch_size=batch_size)

    path_model = "resources/models/20181113.2325/ConvLSTM_model.h5"
    loaded_model = load_model(path_model, custom_objects={'weighted_mse': weighted_mse})
    K.set_value(loaded_model.optimizer.lr, 0.0001)
    """ set callbacks=[tbCallBack] for saving logs to show learning progress in tensorboard """
    history = conv_lstm_model.fit_generator(generator=training_generator, epochs=epochs,
                              validation_data=validation_generator,
                              verbose=1, callbacks=[tensorboard_callback, model_checkpoint_callback], shuffle=False)

    # history = conv_lstm_model.fit_generator(generator=training_generator, epochs=epochs,
    #                               validation_data=validation_generator,
    #                               verbose=1, callbacks=[tensorboard_callback, model_checkpoint_callback], shuffle=False)

    # visualization.visualize_training_progress(history)

    # SAVE MODEL
    conv_lstm_model.save(paths_MODEL)
    print("Model is saved in ", paths_MODEL)
    # conv_lstm_model.summary()


def normalize(movies):
    movies /= 255.0
    # movies -= 0.5
    movies /= 2.0
    movies += 0.25
    return movies


class DataGenerator(Sequence):

    def __init__(self, path, params, batch_size):
        """ sequence_list is list of paths to all sequences (folders) of images"""
        self.sequence_list = get_sequence_list(path)
        np.random.shuffle(self.sequence_list)
        # TODO SORTED ???
        # self.sequence_list.sort()
        self.batch_size = batch_size
        """ params are MoviesParams """
        self.params = params

    def __len__(self):
        return int(np.ceil(len(self.sequence_list) / float(self.batch_size)))

    def __getitem__(self, idx):
        """ Generate one batch of data """
        sequence_list_batch = self.sequence_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = self.__generate_batch(sequence_list_batch)
        return x, y

    # for sigmoid
    # def __normalize_output(self, movies):
    #     movies /= 255.0
    #     movies /= 2.0
    #     movies += 0.25
    #     return movies

    # for relu
    def __normalize_output(self, movies):
        movies[::, ::, ::, 0] = movies[::, ::, ::, 0] / 255.0
        # movies -= 0.5
        movies[::, ::, ::, 0] = movies[::, ::, ::, 0] / 2.0  # do not want to stretch masks
        movies[::, ::, ::, 0] += 0.25
        return movies

    # def __normalize(self, movies):
    #     movies /= 255.0
    #     movies /= 4.0
    #     movies += 0.375
    #     return movies

    def __normalize_input(self, movies):
        movies /= 255.0
        # movies -= 0.5
        movies /= 2.0
        movies += 0.25
        return movies

    def __normalize(self, movies):
        movies /= 255.0
        # movies -= 0.5
        movies /= 2.0
        movies += 0.25
        return movies

    # def __get_mask(self, input_image, output_image):
    #     binary_input_image = to_binary(input_image, 0.01)
    #     binary_output_image = to_binary(output_image, 0.01)
    #
    #     intersection = binary_input_image & binary_output_image
    #     xor = binary_input_image ^ binary_output_image
    #     complement = np.invert(intersection + xor)
    #     # TODO
    #     mask = 255*xor + 0*intersection + 0*complement
    #     return mask

    def __get_mask(self, images):
        images /= 255.0
        mask = to_binary(images[0, ::, ::, 0], 0.01)
        for i in range(1, len(images)-1):
            binary_image = to_binary(images[i+1, ::, ::, 0], 0.01)

            mask = mask | binary_image

            # xor = mask & binary_image
            # mask = intersection + xor
            # plt.imshow(mask)
            # xor = images[i, ::, ::, 0] ^ images[i+1, ::, ::, 0]
            # complement = np.invert(intersection + xor)
            # TODO
            # mask = 255*xor + 0*intersection + 0*complement
        mask_converted = mask.astype(np.float)
        mask_converted /= 1.25
        mask_converted += 0.2
        # mask_uint8 *= 255
        # im = Image.fromarray(mask_uint8)
        # im.show()
        return mask_converted

    def __generate_batch(self, batch_sequence_list):
        n_batch_sequence_list = len(batch_sequence_list)
        initial_movies = np.zeros((n_batch_sequence_list, self.params.frames_ratio.n_frames_in, self.params.frame_dims.rows, self.params.frame_dims.cols, 1), dtype=np.float)  # model input
        """ shifted movies are images shifted by one from initial movies; concatenated with masks"""
        shifted_movies = np.zeros((n_batch_sequence_list, self.params.frame_dims.rows, self.params.frame_dims.cols, 2), dtype=np.float)  # model output

        for i_sequence in range(n_batch_sequence_list):
                """ batch_sequence_list[i_sequence] is path to folder of images"""
                path_to_sequence_folder = batch_sequence_list[i_sequence]
                paths_to_images = glob.glob(path_to_sequence_folder + os.sep + paths.ALL_PNG)  # get paths to all images in the folder
                paths_to_images.sort()
                # images = np.zeros((self.params.frames_ratio.n_frames, self.params.frame_dims.rows, self.params.frame_dims.cols, 1), dtype=np.float)
                """ t is for time """
                for t in range(self.params.frames_ratio.n_frames_in):
                    # print("paths_to_images[t]")
                    # print(paths_to_images[t])
                    # print("paths_to_images[t+1]")
                    # print(paths_to_images[t+1])
                    input_image = np.array(Image.open(paths_to_images[t]))  # initial image

                    # output_image = np.array(Image.open(paths_to_images[t+1]))  # shifted image

                    # mask = self.__get_mask(input_image, output_image)
                    # output_image_with_mask = np.concatenate((output_image, mask), axis=1)

                    initial_movies[i_sequence, t, ::, ::, 0] = input_image[:self.params.frame_dims.rows:, :self.params.frame_dims.cols:]
                    # images[t, ::, ::, 0] = input_image[:self.params.frame_dims.rows:,:self.params.frame_dims.cols:]
                    # shifted_movies[i_sequence, t, ::, ::, 0] = output_image_with_mask
                output_image = np.array(Image.open(paths_to_images[self.params.frames_ratio.n_frames_in]))  # ground truth
                shifted_movies[i_sequence, ::, ::, 0] = output_image[:self.params.frame_dims.rows:, :self.params.frame_dims.cols:]
                # TODO mask
                # images[-1, ::, ::, 0] = output_image

                # mask = self.__get_mask(images)
                # shifted_movies[i_sequence, ::, ::, 1] = mask
        initial_movies = self.__normalize_input(initial_movies)
        shifted_movies = self.__normalize_output(shifted_movies)
        return initial_movies, shifted_movies


# def get_sequence_list(path):
#     sequence_list = []
#     folders_paths = paths.get_all_subfolders(path)
#     for i_folder in range(len(folders_paths)):
#         subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
#         sequence_list.extend(subfolders_paths)
#     return sequence_list

# for randomly generated squares withless complicated folder structure
def get_sequence_list(path):
    # sequence_list = []
    folders_paths = paths.get_all_subfolders(path)
    # for i_folder in range(len(folders_paths)):
    #     subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
    #     sequence_list.extend(subfolders_paths)
    return folders_paths

# # INTERMEDIATE LAYER PREDICTION
# layer_name = "conv_lst_m2d_3"
# intermediate_layer_model = Model(inputs=conv_lstm_model.input, outputs=conv_lstm_model.get_layer(layer_name).output)
# # intermediate_layer_model.summary()
#
# model.predict_intermediate_layer(1000, 1001, intermediate_layer_model, data_obj)
#
#
#
# # PREDICTION
# model.predict_n_random_images(n_predictions=1, from_image=1000, to_image=1001, model=conv_lstm_model, data_obj=data_obj)
# model.predict_images(from_image=1000, to_image=1001, model=conv_lstm_model, data_obj=data_obj)
# del conv_lstm_model
#
#
# # LOAD MODEL
# loaded_model = load_model(paths.MODEL)
# model.predict_images(from_image=1000, to_image=1001, model=loaded_model, data_obj=data_obj)
#
#
# # score, acc = model.evaluate(x_test, y_test,
# #                             batch_size=batch_size)
# # print('Test score:', score)
# # print('Test accuracy:', acc)
#


def print_tensorboard_log_info():
    print("---------------------------------")
    print("Logs from training process will be saved into: ", paths.LOGS)
    print("To see training process: ")
    print(" 1. Open virtual environment: 'source activate env_nowcasting'")
    print(" 2. Call: tensorboard --logdir=%s (with absolute path)" % paths.LOGS)
    print("---------------------------------")


if __name__ == '__main__':
    main()
