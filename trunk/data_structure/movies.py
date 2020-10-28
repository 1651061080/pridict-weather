import glob
import os

import numpy as np
import pylab as plt
from PIL import Image
import tensorflow as tf
from model_structure.helpers import paths
def get_movies_params():
    frame_dims = FrameDims(rows=64, cols=64)
    frames_ratio = FramesRatio(n_frames_in=9, n_frames_out=1)
    # TODO n_samples_processing
    samples_ratio = SamplesRatio(n_samples_processing=0, step=5)
    movies_params = MoviesParams(frame_dims=frame_dims, frames_ratio=frames_ratio, samples_ratio=samples_ratio)
    return movies_params


class MoviesTuple:
    def __init__(self, path, movies_params):
        self.movies_params = movies_params
        self.path = path
        """ input movies """
        self.initial_movies = None
        """ target (output) movies """
        self.shifted_movies = None

    def get_input_training_data(self):
        """
        Get input data for training limited by n_samples_processing
        :return: input data for training
        """
        return self.initial_movies.get_samples_processing()

    def get_target_training_data(self):
        """
        Get target (output) data for training limited by n_samples_processing
        :return: target (output) data for training, shifted training input images by one frame
        """
        return self.shifted_movies.get_samples_processing()

    def normalize(self):
        self.initial_movies.normalize()
        self.shifted_movies.normalize()

    def load_movies(self):
        self.set_n_samples_loaded()
        self.initial_movies = Movies(path=self.path, movies_params=self.movies_params)
        self.shifted_movies = Movies(path=self.path, movies_params=self.movies_params)
        # TODO n_samples processing or loaded?
        self.initial_movies.movies = np.zeros((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in, self.movies_params.frame_dims.rows, self.movies_params.frame_dims.cols, 1), dtype=np.float)
        self.shifted_movies.movies = np.zeros((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in, self.movies_params.frame_dims.rows, self.movies_params.frame_dims.cols, 1), dtype=np.float)
        self.initial_movies.names = np.empty((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in), dtype=object)
        self.shifted_movies.names = np.empty((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in), dtype=object)

        i_samples_laoded = 0  # complex folder structure
        """ 0002, 0000, 0001, ... sorted """
        folders_paths = paths.get_all_subfolders(self.path)
        folders_paths.sort()
        n_folders = len(folders_paths)
        for i_folder in range(n_folders):
            """ row0081_col0041, row0041_col0361, ... sorted """
            subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
            subfolders_paths.sort()
            n_subfolders = len(subfolders_paths)
            for i_subfolder in range(n_subfolders):
                """ list of image in one target folder, like in folder row0001_col0201 """
                image_list, name_list = get_images_from_folder(subfolders_paths[i_subfolder])
                """ number of images in one subfolder """
                image_list_len = len(image_list)
                """ j is index of samples in movies read from one target folder containing images """
                j = 0
                """ i is index of images in one target folder, like in folder row0001_col0201 """
                for i in range(0, image_list_len - self.movies_params.frames_ratio.n_frames_in, self.movies_params.samples_ratio.step):  # -1 because prediction is one step ahead (but shifted by 1, because here indexed from 0, in folders indexed from 1)
                    """ t is for time """
                    for t in range(self.movies_params.frames_ratio.n_frames_in):
                        self.initial_movies.set_frame(i_samples_laoded + j, t, np.array(image_list[i + t]))
                        self.shifted_movies.set_frame(i_samples_laoded + j, t, np.array(image_list[i + t + 1]))
                        self.initial_movies.set_name(i_samples_laoded + j, t, np.array(name_list[i + t]))
                        self.shifted_movies.set_name(i_samples_laoded + j, t, np.array(name_list[i + t + 1]))
                    j += 1
                """ index is index of samples in movies read from all target folders containing images """
                i_samples_laoded += j

    def load_movies_with_weights(self):
        self.set_n_samples_loaded()
        self.initial_movies = Movies(path=self.path, movies_params=self.movies_params)
        self.shifted_movies = Movies(path=self.path, movies_params=self.movies_params)
        # TODO n_samples processing or loaded?
        self.initial_movies.movies = np.zeros((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in, self.movies_params.frame_dims.rows, self.movies_params.frame_dims.cols, 1), dtype=np.float)
        self.shifted_movies.movies = np.zeros((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in, self.movies_params.frame_dims.rows, self.movies_params.frame_dims.cols * 2, 1), dtype=np.float)
        self.initial_movies.names = np.empty((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in), dtype=object)
        self.shifted_movies.names = np.empty((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in), dtype=object)

        i_samples_laoded = 0  # complex folder structure
        """ 0002, 0000, 0001, ... sorted """
        folders_paths = paths.get_all_subfolders(self.path)
        folders_paths.sort()
        n_folders = len(folders_paths)
        for i_folder in range(n_folders):
            """ row0081_col0041, row0041_col0361, ... sorted """
            subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
            subfolders_paths.sort()
            n_subfolders = len(subfolders_paths)
            for i_subfolder in range(n_subfolders):
                """ list of image in one target folder, like in folder row0001_col0201 """
                image_list, name_list = get_images_from_folder(subfolders_paths[i_subfolder])
                """ number of images in one subfolder """
                image_list_len = len(image_list)
                """ j is index of samples in movies read from one target folder containing images """
                j = 0
                """ i is index of images in one target folder, like in folder row0001_col0201 """
                for i in range(0, image_list_len - self.movies_params.frames_ratio.n_frames_in, self.movies_params.samples_ratio.step):  # -1 because prediction is one step ahead (but shifted by 1, because here indexed from 0, in folders indexed from 1)
                    """ t is for time """
                    for t in range(self.movies_params.frames_ratio.n_frames_in):
                        y_true = np.array(image_list[i + t])
                        y_pred = np.array(image_list[i + t + 1])

                        binary_true = to_binary(y_true, 0.01)
                        binary_pred = to_binary(y_pred, 0.01)

                        intersection = binary_true & binary_pred
                        xor = binary_true ^ binary_pred
                        complement = np.invert(intersection + xor)
                        mask = 255*xor + 0*intersection + 0*complement

                        y_pred_with_mask = np.concatenate((y_pred, mask), axis=1)
                        # half_size = int(y_pred_with_mask.shape[1] / 2)
                        # y_true_t = y_pred_with_mask[:, :half_size]
                        # y_pred_t = y_pred_with_mask[:, half_size:]

                        self.initial_movies.set_frame(i_samples_laoded + j, t, y_true)
                        self.shifted_movies.set_frame(i_samples_laoded + j, t, y_pred_with_mask)
                        self.initial_movies.set_name(i_samples_laoded + j, t, np.array(name_list[i + t]))
                        self.shifted_movies.set_name(i_samples_laoded + j, t, np.array(name_list[i + t + 1]))
                    j += 1
                """ index is index of samples in movies read from all target folders containing images """
                i_samples_laoded += j

    def set_n_samples_loaded(self):
        n_samples_laoded = 0
        """ 0002, 0000, 0001, ... """
        """ NOT SORTED !!!! """
        folders_paths = paths.get_all_subfolders(self.path)
        n_folders = len(folders_paths)
        for i_folder in range(n_folders):
            """ row0081_col0041, row0041_col0361, ...  """
            """ NOT SORTED !!!! """
            subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
            n_subfolders = len(subfolders_paths)
            for i_subfolder in range(n_subfolders):
                """ list of images in one target folder, like in folder row0001_col0201 """
                image_list, name_list = get_images_from_folder(subfolders_paths[i_subfolder])
                """ number of images in one subfolder """
                image_list_len = len(image_list)
                """ i is index of images in one target folder, like in folder row0001_col0201 """
                for i in range(0, image_list_len - self.movies_params.frames_ratio.n_frames_in, self.movies_params.samples_ratio.step):  # -1 because prediction is one step ahead (but shifted by 1, because here indexed from 0, in folders indexed from 1)
                    n_samples_laoded += 1
        self.movies_params.samples_ratio.n_samples_loaded = n_samples_laoded


class Movies:
    def __init__(self, path, movies_params):
        self.movies_params = movies_params
        self.path = path
        self.movies = None
        self.names = None

    def normalize(self):
        """ Normalize movies between 0 and 1 """
        print("Normalization: [0,1]")
        self.movies /= self.movies.max()

    def get_sample(self, i_sample: int) -> np.ndarray:  # (samples, frames, rows, cols, 1) -> (frames, rows, cols, 1)
        sample = self.movies[i_sample][::, ::, ::, ::]
        return sample

    def get_frame(self, i_sample, i_frame):
        result = self.movies[i_sample][i_frame, ::, ::, 0]
        return result

    def get_input_frames(self, i_sample: int) -> np.ndarray:
        sample = self.get_sample(i_sample)
        result = sample[:self.movies_params.frames_ratio.n_frames_in, ::, ::, ::]  # (frames, rows, cols, 1) -> (frames, rows, cols, 1)
        return result

    def get_name(self, i_sample, i_frame):
        result = self.names[i_sample, i_frame]
        return result

    def set_frame(self, i_sample: int, i_frame: int, frame_to_set: np.ndarray):  # (samples, frames, rows, cols, 1)
        self.movies[i_sample, i_frame, ::, ::, 0] = frame_to_set

    def set_name(self, i_sample: int, i_frame: int, name_to_set):  # (samples, frames, rows, cols, 1)
        self.names[i_sample, i_frame] = name_to_set.item()

    """ Show normlized image. For different scale, change vmin and vmax. """
    def show_image(self, i_sample, i_frame):
        frame = self.get_frame(i_sample, i_frame)
        plt.figure()
        plt.title(self.get_name(i_sample, i_frame))
        plt.imshow(frame, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

    """ Plot normlized image into figure. For different scale, change vmin and vmax. """
    def plot_image(self, i_sample, i_frame):
        frame = self.get_frame(i_sample, i_frame)
        plt.title(self.get_name(i_sample, i_frame))
        plt.imshow(frame, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

    def get_samples_processing(self):
        return self.movies[:self.movies_params.samples_ratio.n_samples_processing]

    def load_movies(self):
        self.set_n_samples_loaded()
        self.movies = np.zeros((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_out, self.movies_params.frame_dims.rows, self.movies_params.frame_dims.cols, 1), dtype=np.float)
        self.names = np.empty((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_out), dtype=object)

        folders_paths = paths.get_all_subfolders(self.path)
        folders_paths.sort()
        n_folders = len(folders_paths)
        for i_folder in range(n_folders):
            subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
            subfolders_paths.sort()
            n_subfolders = len(subfolders_paths)
            for i_subfolder in range(n_subfolders):
                image_list, name_list = get_images_from_folder(subfolders_paths[i_subfolder])
                i_sample = i_folder * n_subfolders + i_subfolder
                self.set_frame(i_sample, 0, np.array(image_list[-1]))
                self.set_name(i_sample, 0, np.array(name_list[-1]))

    def set_n_samples_loaded(self):
        n_samples_laoded = 0
        folders_paths = paths.get_all_subfolders(self.path)
        n_folders = len(folders_paths)
        for i_folder in range(n_folders):
            subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
            n_subfolders = len(subfolders_paths)
            for i_subfolder in range(n_subfolders):
                n_samples_laoded += 1
        self.movies_params.samples_ratio.n_samples_loaded = n_samples_laoded

    def load_movies_loss_weights(self):
        """ loading data masks for loss_weights """
        self.set_n_samples_loaded_for_loss_weights()
        # TODO n_samples processing or loaded?
        self.movies = np.zeros((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in, self.movies_params.frame_dims.rows, self.movies_params.frame_dims.cols, 1), dtype=np.float)
        self.names = np.empty((self.movies_params.samples_ratio.n_samples_loaded, self.movies_params.frames_ratio.n_frames_in), dtype=object)

        i_samples_laoded = 0  # complex folder structure
        """ 0002, 0000, 0001, ... sorted """
        folders_paths = paths.get_all_subfolders(self.path)
        folders_paths.sort()
        n_folders = len(folders_paths)
        for i_folder in range(n_folders):
            """ row0081_col0041, row0041_col0361, ... sorted """
            subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
            subfolders_paths.sort()
            n_subfolders = len(subfolders_paths)
            for i_subfolder in range(n_subfolders):
                """ list of image in one target folder, like in folder row0001_col0201 """
                image_list, name_list = get_images_from_folder(subfolders_paths[i_subfolder])
                """ number of images in one subfolder """
                image_list_len = len(image_list)
                """ j is index of samples in movies read from one target folder containing images """
                j = 0
                """ i is index of images in one target folder, like in folder row0001_col0201 """
                for i in range(0, image_list_len - self.movies_params.frames_ratio.n_frames_in + 1, self.movies_params.samples_ratio.step):  # -1 because prediction is one step ahead (but shifted by 1, because here indexed from 0, in folders indexed from 1)
                    """ t is for time """
                    for t in range(self.movies_params.frames_ratio.n_frames_in):
                        self.set_frame(i_samples_laoded + j, t, np.array(image_list[i + t]))
                        self.set_name(i_samples_laoded + j, t, np.array(name_list[i + t]))
                    j += 1
                """ index is index of samples in movies read from all target folders containing images """
                i_samples_laoded += j

    def set_n_samples_loaded_for_loss_weights(self):
        """ compute number of mask samples for loss_weights """
        n_samples_laoded = 0
        """ 0002, 0000, 0001, ... """
        """ NOT SORTED !!!! """
        folders_paths = paths.get_all_subfolders(self.path)
        n_folders = len(folders_paths)
        for i_folder in range(n_folders):
            """ row0081_col0041, row0041_col0361, ...  """
            """ NOT SORTED !!!! """
            subfolders_paths = paths.get_all_subfolders(folders_paths[i_folder])
            n_subfolders = len(subfolders_paths)
            for i_subfolder in range(n_subfolders):
                """ list of images in one target folder, like in folder row0001_col0201 """
                image_list, name_list = get_images_from_folder(subfolders_paths[i_subfolder])
                """ number of images in one subfolder """
                image_list_len = len(image_list)
                """ i is index of images in one target folder, like in folder row0001_col0201 """
                for i in range(0, image_list_len - self.movies_params.frames_ratio.n_frames_in + 1, self.movies_params.samples_ratio.step):  # -1 because prediction is one step ahead (but shifted by 1, because here indexed from 0, in folders indexed from 1)
                    n_samples_laoded += 1
        self.movies_params.samples_ratio.n_samples_loaded = n_samples_laoded


def get_last_frame(array_5D: np.ndarray) -> np.ndarray:  # (samples, frames, rows, cols, 1) -> (1(last_frame), rows, cols, 1)
    result = array_5D[::, -1, ::, ::, ::]
    return result


class MoviesParams:
    def __init__(self, frame_dims, frames_ratio, samples_ratio):
        self.frame_dims = frame_dims
        self.frames_ratio = frames_ratio
        self.samples_ratio = samples_ratio


class FrameDims:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols


class FramesRatio:
    def __init__(self, n_frames_in, n_frames_out):
        self.n_frames_in = n_frames_in
        self.n_frames_out = n_frames_out
        self.n_frames = n_frames_in + n_frames_out


class SamplesRatio:
    def __init__(self, n_samples_processing, step):
        self.step = step
        """ n_samples_processing is integer for limiting number of samples for training,
            if you do not want to train the model on all images in a folder"""
        self.n_samples_processing = n_samples_processing
        self.n_samples_loaded = 0
        # if self.n_samples_processing > self.n_samples_loaded:
        #     raise Exception('n_samples_processing should be <= than n_samples_loaded')


def get_images_from_folder(folder_path):
    image_list = []
    name_list = []
    paths_to_images = glob.glob(folder_path + os.sep + paths.ALL_PNG)
    """ sort byt datetime"""
    paths_to_images.sort()
    for path_to_image in paths_to_images:
        image = Image.open(path_to_image)
        image_list.append(image)
        name_list.append(os.path.basename(path_to_image))
    return image_list, name_list


def to_binary(img, trashold):
    return trashold < img
