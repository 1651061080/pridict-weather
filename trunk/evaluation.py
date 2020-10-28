import tensorflow as tf
from keras.engine.saving import load_model

from data_structure import movies as ms
from helpers import paths
from model_structure.model_convLSTM import weighted_mse
from training import DataGenerator


def main():
    # datetime_folder = input("Enter name (datetime) of folder containing predictions located in resources/prediction (format: 20180910.0817)" + os.linesep) + os.sep
    # model_folder = input("Enter name of subfolder containing prediction for model for specific epoch located in resources/prediction/datetime (format: model.epoch-03-val_loss-0.0025.hdf5)" + os.linesep) + os.sep
    # prediction_folder = os.path.join(paths.PREDICTION, datetime_folder, model_folder)

    # change in paths what data evaluate
    path_model = '/home/vladka/Desktop/DP/Projects/xhezelov_nowcasting/trunk/resources/models/20181122.1313/model.epoch-100-val_loss-0.0004.hdf5'

    batch_size = 1
    validation_generator = DataGenerator(path=paths.DATA_FOR_TESTING, params=ms.get_movies_params(), batch_size=batch_size)
    loaded_model = load_model(path_model, custom_objects={'weighted_mse': weighted_mse})
    test_loss = loaded_model.evaluate_generator(generator=validation_generator)

    print(test_loss)


def print_mse_of_two_sets(first_set, second_set):
    """ Mean squared error """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mse = tf.reduce_sum(tf.square(first_set - second_set))
    print("Mean squared error", sess.run(mse))


if __name__ == '__main__':
    main()
