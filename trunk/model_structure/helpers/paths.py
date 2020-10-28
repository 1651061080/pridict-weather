import datetime
import os
import glob
import re

RESOURCES = "resources" + os.sep
__radar_images = "radar_images" + os.sep
__ground_truth = "ground_truth" + os.sep
__cropped = "cropped_40x40" + os.sep

# DATA_FOR_TRAINING = os.path.join(RESOURCES, __radar_images, __ground_truth, __cropped)
__radar_data = "radar_data" + os.sep
__training_data = "training_data" + os.sep
__training_data_90x90 = "training_data_final" + os.sep

__synthetic_data = "synthetic_data" + os.sep
__synthetic_data_type = "10_radar_binary" + os.sep
#tyfon_training = "/cygdrive/c/Projects/Radar Prediction/data_preprocessing/data/06_patches/training_dense/"
#tyfon_training = "C:" + os.sep + "Projects" + os.sep + "Radar Prediction" + os.sep + "data_preprocessing" + os.sep + "data" + os.sep + "06_patches" + os.sep + "training_dense" + os.sep
#DATA_FOR_TRAINING = tyfon_training
#DATA_FOR_TRAINING = os.path.join(RESOURCES, __synthetic_data, __synthetic_data_type)
DATA_FOR_TRAINING = os.path.join(RESOURCES, __radar_data, __training_data, __training_data_90x90)

__synthetic_data_validation = "synthetic_data_validation" + os.sep

__validation_data = "validation_data" + os.sep
__validation_data_90x90 = "validation_data_20181201" + os.sep
#tyfon_validation = "C:" + os.sep + "Projects" + os.sep + "Radar Prediction" + os.sep + "data_preprocessing" + os.sep + "data" + os.sep + "06_patches" + os.sep + "validation" + os.sep
#DATA_FOR_VALIDATION = tyfon_validation
#DATA_FOR_VALIDATION = os.path.join(RESOURCES, __synthetic_data_validation, __synthetic_data_type)
DATA_FOR_VALIDATION = os.path.join(RESOURCES, __radar_data, __validation_data, __validation_data_90x90)


__testing_data = "testing_data" + os.sep
__testing_data_90x90 = "06_patches_fullFrames" + os.sep + "20180728.1220" + os.sep
DATA_FOR_TESTING = os.path.join(RESOURCES, __radar_data, __testing_data, __testing_data_90x90)

__prediction_chmu_validation = "prediction_chmu_validation" + os.sep
__prediction_chmu_validation_90x90 = "prediction_chmu_validation_90x90" + os.sep
PREDICTION_CHMU_VALIDATION = os.path.join(RESOURCES, __radar_data, __prediction_chmu_validation, __prediction_chmu_validation_90x90)

__masks_for_weighted_loss = "training_data_90x90_ternary_masks" + os.sep
MASKS = os.path.join(RESOURCES, __radar_data, __training_data, __masks_for_weighted_loss)

# RESOURCES = "resources" + os.sep
# __data_for_learning = "data_for_learning" + os.sep
# DATA_FOR_LEARNING = os.path.join(RESOURCES, __data_for_learning)

ALL_PNG = "*.png"

__prediction = "prediction" + os.sep
PREDICTION = os.path.join(RESOURCES, __prediction)

__prediction_strips = "prediction_strips" + os.sep
PREDICTION_STRIPS = os.path.join(RESOURCES, __prediction_strips)


__models = "models" + os.sep
MODELS = os.path.join(RESOURCES, __models)


now = datetime.datetime.now().strftime('%Y%m%d.%H%M')


def create_folder_with_timestamp(path):
    folder = path + now
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder + os.sep


__visualization = "visualization" + os.sep
VISUALIZATION = os.path.join(RESOURCES, __visualization)
MODEL_GRAPH = os.path.join(VISUALIZATION, "model_graph.png")
TRAINING_PROGRESS = os.path.join(VISUALIZATION, "training_progress.png")

__logs = "logs_tensorboard" + os.sep
LOGS = os.path.join(RESOURCES, __logs)

__intermediate_layer = "intermediate_layer" + os.sep
INTERMEDIATE_LAYER = os.path.join(RESOURCES, __visualization, __intermediate_layer)

ZERO_PADDING = 4  # pad numbers (numbers in folders names) with ZERO_PADDING zeros


# move to folder_structure
def get_all_subfolders(folder_path):
    return glob.glob(folder_path + "*" + os.sep)


def gel_all_models(folder_path):
    return glob.glob(folder_path + "*.hdf5")


def get_path_without_extension(path):
    return os.path.splitext(path)[0]


def get_datetime_from_name(name):
    reg = re.search('\d+\.\d+', name)
    return reg[0]


def get_name(name):
    reg = re.search('\d+\.\d+__\d+x\d+', name)
    return reg[0]
#
# def get_name(name):
#     reg = re.search('(.*)(\.)', name)
#     return reg[0]
#
# def get_folder_for_prediction(name):
#     reg = re.search('(.*)', name)
#     return reg[0]


def get_row_col_from_name(name):
    reg = re.search('row\d+_col\d+', name)
    return reg[0]


