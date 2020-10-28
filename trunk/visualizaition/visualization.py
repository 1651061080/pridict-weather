import os

import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.widgets import Button

from data_structure import movies as ms
from helpers import paths


def main():

    datetime_folder = input("Enter name (datetime) of folder containing predictions located in resources/prediction (format: 20180910.0817)" + os.linesep) + os.sep
    # datetime_folder = '20181025.2126' + os.sep

    model_folder = input("Enter name of subfolder containing prediction for model for specific epoch located in resources/prediction/datetime (format: model.epoch-03-val_loss-0.0025.hdf5)" + os.linesep) + os.sep
    # model_folder = 'model.epoch-02-val_loss-0.0008.hdf5' + os.sep

    prediction_folder = os.path.join(paths.PREDICTION, datetime_folder, model_folder)

    validation_movies_tuple = ms.MoviesTuple(path=paths.DATA_FOR_VALIDATION, movies_params=ms.get_movies_params())
    validation_movies_tuple.load_movies()
    validation_movies_tuple.normalize()

    prediction_movies_nn = ms.Movies(path=prediction_folder, movies_params=ms.get_movies_params())
    prediction_movies_nn.load_movies()
    prediction_movies_nn.normalize()

    prediction_movies_chmu = ms.Movies(path=paths.PREDICTION_CHMU_VALIDATION, movies_params=ms.get_movies_params())
    prediction_movies_chmu.load_movies()
    prediction_movies_chmu.normalize()
    # TODO chmu predictionvalidation_movies_tuple

    show_comparison(validation_movies_tuple, prediction_movies_nn, prediction_movies_chmu)


def show_comparison(validation_movies_tuple, prediction_movies_nn, prediction_movies_chmu):
    fig = plt.figure(figsize=(15, 7))

    frame_tracker = FrameTracker(fig, validation_movies_tuple, prediction_movies_nn, prediction_movies_chmu)
    sample_tracker = SampleTracker(n_sapmles=validation_movies_tuple.movies_params.samples_ratio.n_samples_loaded, frame_tracker=frame_tracker)

    """ Buttons for navigation among samples """
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(sample_tracker.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(sample_tracker.prev)

    fig.canvas.mpl_connect('scroll_event', frame_tracker.onscroll)
    # fig.canvas.mpl_connect('button_press_event', sample_tracker.next)
    # fig.canvas.mpl_connect('button_press_event', sample_tracker.prev)

    plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.show()


def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


class SampleTracker(object):

    def __init__(self, n_sapmles, frame_tracker):
        self.n_sapmles = n_sapmles
        self.frame_tracker = frame_tracker

    def next(self, event):
        print("Sample %d" % self.frame_tracker.i_sample)
        self.frame_tracker.i_sample = (self.frame_tracker.i_sample + 1) % self.n_sapmles
        self.frame_tracker.i_frame = 0
        self.frame_tracker.update()
        plt.draw()

    def prev(self, event):
        print("Sample %d" % self.frame_tracker.i_sample)
        self.frame_tracker.i_sample = (self.frame_tracker.i_sample - 1) % self.n_sapmles
        self.frame_tracker.i_frame = 0
        self.frame_tracker.update()
        plt.draw()


class FrameTracker(object):
    def __init__(self, fig, validation_movies_tuple, prediction_movies_nn, prediction_movies_chmu):
        self.fig = fig
        self.fig.add_subplot(131)
        self.fig.add_subplot(132)
        self.fig.add_subplot(133)
        self.fig.text(0.01, 0.01, 'Use scroll wheel to navigate images', fontsize=8)

        allaxes = fig.get_axes()
        self.validation_movies_tuple = validation_movies_tuple
        self.prediction_movies_nn = prediction_movies_nn
        self.prediction_movies_chmu = prediction_movies_chmu
        self.i_sample = 2
        self.sample = self.validation_movies_tuple.initial_movies.get_sample(self.i_sample)

        self.ax_ground_truth = allaxes[0]
        self.ax_my_prediction = allaxes[1]
        self.ax_chmu_prediction = allaxes[2]

        self.slices = validation_movies_tuple.movies_params.frames_ratio.n_frames
        self.i_frame = 0

        self.frame_ground_truth = validation_movies_tuple.initial_movies.get_frame(self.i_sample, self.i_frame)

        self.im_ground_truth = self.ax_ground_truth.imshow(self.frame_ground_truth, cmap=plt.get_cmap('gray'))
        self.im_my_prediction = self.ax_my_prediction.imshow(self.frame_ground_truth, cmap=plt.get_cmap('gray'))
        self.im_chmu_prediction = self.ax_chmu_prediction.imshow(self.frame_ground_truth, cmap=plt.get_cmap('gray'))

        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.i_frame = (self.i_frame + 1) % self.slices
        else:
            self.i_frame = (self.i_frame - 1) % self.slices
        self.update()

    def update(self):
        print("Sample in update %d" % self.i_sample)
        if self.i_frame < self.validation_movies_tuple.movies_params.frames_ratio.n_frames_in:

            self.frame_ground_truth = self.validation_movies_tuple.initial_movies.get_frame(self.i_sample, self.i_frame)

            self.im_ground_truth.set_data(self.frame_ground_truth)
            self.im_my_prediction.set_data(self.frame_ground_truth)
            self.im_chmu_prediction.set_data(self.frame_ground_truth)

            self.fig.suptitle('INPUT' + os.linesep + 'Frame %d' % (self.i_frame + 1), fontsize=12)

            self.ax_ground_truth.set_title("Ground Truth")
            self.ax_ground_truth.set_xlabel(self.validation_movies_tuple.initial_movies.get_name(self.i_sample, self.i_frame))

            self.ax_my_prediction.set_title("NN Prediction")
            self.ax_my_prediction.set_xlabel(self.validation_movies_tuple.initial_movies.get_name(self.i_sample, self.i_frame))

            self.ax_chmu_prediction.set_title("CHMU Prediction")
            self.ax_chmu_prediction.set_xlabel(self.validation_movies_tuple.initial_movies.get_name(self.i_sample, self.i_frame))

            plt.draw()

        """ prediction frame """
        if self.i_frame == self.validation_movies_tuple.movies_params.frames_ratio.n_frames_in:

            self.frame_ground_truth = self.validation_movies_tuple.shifted_movies.get_frame(self.i_sample, self.i_frame-1)
            frame_my_prediction = self.prediction_movies_nn.get_frame(self.i_sample, 0)
            frame_chmu_prediction = self.prediction_movies_chmu.get_frame(self.i_sample, 0)

            self.im_ground_truth.set_data(self.frame_ground_truth)
            self.im_my_prediction.set_data(frame_my_prediction)
            self.im_chmu_prediction.set_data(frame_chmu_prediction)

            self.fig.suptitle('PREDICTION' + os.linesep + 'Frame %d' % (self.i_frame + 1), fontsize=12)

            self.ax_ground_truth.set_title("Ground Truth")
            self.ax_ground_truth.set_xlabel(self.validation_movies_tuple.shifted_movies.get_name(self.i_sample, self.i_frame-1))

            mse_nn = get_mse_of_two_frames(self.frame_ground_truth, frame_my_prediction)
            self.ax_my_prediction.set_title(r"$\bf{" + "MSE = " + str(round(mse_nn, 2)) + "}$" + os.linesep + "NN Prediction")
            self.ax_my_prediction.set_xlabel(self.prediction_movies_nn.get_name(self.i_sample, 0))

            mse_chmu = get_mse_of_two_frames(self.frame_ground_truth, frame_chmu_prediction)
            self.ax_chmu_prediction.set_title(r"$\bf{" + "MSE = " + str(round(mse_chmu, 2)) + "}$" + os.linesep + "CHMU Prediction")
            self.ax_chmu_prediction.set_xlabel(self.prediction_movies_chmu.get_name(self.i_sample, 0))

            plt.draw()


def get_mse_of_two_frames(first_frame, second_frame):
    """ Mean squared error """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mse = tf.reduce_sum(tf.square(first_frame - second_frame))
    result = sess.run(mse)
    return result


if __name__ == '__main__':
    main()


def visualize_training_progress(history):

    plt.figure(figsize=(12, 7))
    plt.figure(1)
    plt.subplot(211)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.subplot(212)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.tight_layout()  # to not overlay labels

    plt.savefig(paths.TRAINING_PROGRESS)


class RadarColorMap:

    def __init__(self):

        self.__RADAR_COLOR_MAP_MATRIX = [
            [0.0, 0.0, 0.0],
            [0.2196, 0.0, 0.4392],
            [0.1882, 0.0, 0.6588],
            [0.0, 0.0, 0.9882],
            [0.0, 0.4235, 0.7529],
            [0.0, 0.6275, 0.0],
            [0.0, 0.7373, 0.0],
            [0.2039, 0.8471, 0.0],
            [0.6118, 0.8627, 0.0],
            [0.8784, 0.8627, 0.0],
            [0.9882, 0.6902, 0.0],
            [0.9882, 0.5176, 0.0],
            [0.9882, 0.3451, 0.0],
            [0.9882, 0.0, 0.0],
            [0.6275, 0.0, 0.0],
            [0.9882, 0.9882, 0.9882]]
        self.radar_color_map = col.ListedColormap(self.__RADAR_COLOR_MAP_MATRIX[0:15], 'indexed')
        cm.register_cmap(cmap=self.radar_color_map)

