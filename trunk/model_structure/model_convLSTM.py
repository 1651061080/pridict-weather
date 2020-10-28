from keras import optimizers
from keras.layers import *
from keras.losses import mean_squared_error
from keras.models import *
from keras.optimizers import *

from data_structure import movies as ms


def get_model():

    movies_params = ms.get_movies_params()
    rows = movies_params.frame_dims.rows
    cols = movies_params.frame_dims.cols
    inputs = Input(shape=(None, rows, cols, 1))

    # ARCHITECTURE UNITOS4
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))(conv1)

    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))(conv2)

    conv3 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))(conv3)

    conv4 = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(pool3)
    conv4 = BatchNormalization()(conv4)

    up5 = TimeDistributed(UpSampling2D(size=(2, 2), data_format=None))(conv4)
    conv5 = TimeDistributed(Conv2D(filters=128, kernel_size=(2, 2), activation='sigmoid', padding='same', data_format='channels_last'))(up5)
    conv5 = BatchNormalization()(conv5)

    merge5 = concatenate([conv3, conv5])
    conv6 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(merge5)
    conv6 = BatchNormalization()(conv6)

    up7 = TimeDistributed(UpSampling2D(size=(2, 2), data_format=None))(conv6)
    conv7 = TimeDistributed(Conv2D(filters=64, kernel_size=(2, 2), activation='sigmoid', padding='same', data_format='channels_last'))(up7)
    conv7 = BatchNormalization()(conv7)

    merge7 = concatenate([conv2, conv7])
    conv8 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(merge7)
    conv8 = BatchNormalization()(conv8)

    up9 = TimeDistributed(UpSampling2D(size=(2, 2), data_format=None))(conv8)
    conv9 = TimeDistributed(Conv2D(filters=32, kernel_size=(2, 2), activation='sigmoid', padding='same', data_format='channels_last'))(up9)
    conv9 = BatchNormalization()(conv9)

    merge9 = concatenate([conv1, conv9])
    conv10 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, return_sequences=False)(merge9)
    conv10 = BatchNormalization()(conv10)

    outputs = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last')(conv10)

    # ARCHITECTURE UNITOS3
    #
    # conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(inputs)
    # conv1 = BatchNormalization()(conv1)
    # pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))(conv1)
    #
    # conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(pool1)
    # conv2 = BatchNormalization()(conv2)
    # pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))(conv2)
    #
    # conv3 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(pool2)
    # conv3 = BatchNormalization()(conv3)
    #
    # up4 = TimeDistributed(UpSampling2D(size=(2, 2), data_format=None))(conv3)
    # conv4 = TimeDistributed(Conv2D(filters=64, kernel_size=(2, 2), activation='sigmoid', padding='same', data_format='channels_last'))(up4)
    # conv4 = BatchNormalization()(conv4)
    #
    # merge4 = concatenate([conv2, conv4])
    # conv5 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', strides=1, return_sequences=True)(merge4)
    # conv5 = BatchNormalization()(conv5)
    #
    # up6 = TimeDistributed(UpSampling2D(size=(2, 2), data_format=None))(conv5)
    # conv6 = TimeDistributed(Conv2D(filters=32, kernel_size=(2, 2), activation='sigmoid', padding='same', data_format='channels_last'))(up6)
    # conv6 = BatchNormalization()(conv6)
    #
    # merge6 = concatenate([conv1, conv6])
    # conv7 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', strides=1, return_sequences=False)(merge6)
    # conv7 = BatchNormalization()(conv7)
    #
    # outputs = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_last')(conv7)

    # ARCHITECTURE PRIMITOS
    #
    # layer_1_filters = 16
    # layer_1_kernel_size = (3, 3)
    # layer_1_strides = (1, 1)
    #
    # outputs = ConvLSTM2D(filters=layer_1_filters, kernel_size=layer_1_kernel_size,
    #                      padding='same', strides=layer_1_strides, return_sequences=True)(inputs)
    #
    # print("---------------------------------")
    # print("---------------------------------")
    # print("ConvLSTM2D:")
    # print("layer_1_filters = ", layer_1_filters)
    # print("layer_1_kernel_size = ", layer_1_kernel_size)
    # print("layer_1_strides = ", layer_1_strides)
    #
    # outputs = BatchNormalization()(outputs)
    #
    # layer_2_filters = 32
    # layer_2_kernel_size = (5, 5)
    # layer_2_strides = (1, 1)
    #
    # outputs = ConvLSTM2D(filters=layer_2_filters, kernel_size=layer_2_kernel_size,
    #                      padding='same', strides=layer_2_strides, return_sequences=True)(outputs)
    #
    # print("---------------------------------")
    # print("ConvLSTM2D:")
    # print("layer_2_filters = ", layer_2_filters)
    # print("layer_2_kernel_size = ", layer_2_kernel_size)
    # print("layer_2_strides = ", layer_2_strides)
    #
    # outputs = BatchNormalization()(outputs)
    #
    # layer_3_filters = 64
    # layer_3_kernel_size = (9, 9)
    # layer_3_strides = (1, 1)
    #
    # outputs = ConvLSTM2D(filters=layer_3_filters, kernel_size=layer_3_kernel_size,
    #                      padding='same', strides=layer_3_strides, return_sequences=False)(outputs)
    #
    # print("---------------------------------")
    # print("ConvLSTM2D:")
    # print("layer_3_filters = ", layer_3_filters)
    # print("layer_3_kernel_size = ", layer_3_kernel_size)
    # print("layer_3_strides = ", layer_3_strides)
    #
    # outputs = BatchNormalization()(outputs)
    # #
    # # outputs = ConvLSTM2D(filters=32, kernel_size=(3, 3),
    # #                      padding='same', strides=layer_3_strides, return_sequences=False)(outputs)
    # #
    # # print("---------------------------------")
    # # print("ConvLSTM2D:")
    # # print("layer_3_filters = ", layer_3_filters)
    # # print("layer_3_kernel_size = ", layer_3_kernel_size)
    # # print("layer_3_strides = ", layer_3_strides)
    # #
    # # outputs = BatchNormalization()(outputs)
    #
    # layer_4_filters = 1
    # layer_4_kernel_size = (3, 3)
    # activation = 'sigmoid'
    #
    # outputs = Conv2D(filters=layer_4_filters, kernel_size=layer_4_kernel_size,
    #                  activation=activation,
    #                  padding='same', data_format='channels_last')(outputs)
    #
    # print("---------------------------------")
    # print("Conv3D:")
    # print("layer_4_filters = ", layer_4_filters)
    # print("layer_4_kernel_size = ", layer_4_kernel_size)
    # print("Activation: ", activation)

    model_conv_lstm = Model(inputs, outputs)

    loss = "mean_squared_error"
    # optimizer = 'adadelta'
    # optimizer = 'rmsprop'
    # optimizer = optimizers.RMSProp(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
    optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    metrics = ['accuracy', 'mean_squared_error']

    # model_conv_lstm.compile(loss=my_mean_squared_error, optimizer=optimizer, metrics=metrics)
    # model_conv_lstm.compile(loss=[weighted_mse(size=rows, padding=1)], optimizer=optimizer)
    # model_conv_lstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[weighted_mse])

    model_conv_lstm.compile(loss=weighted_mse, optimizer=optimizer)

    print("---------------------------------")
    print("Loss: ", loss)
    print("Optimizer: ", optimizer)
    print("Metrics: ", metrics)
    print("Padding is 'same' in every layer")
    print("---------------------------------")
    print("---------------------------------")

    """
    Write summary of model
    """
    model_conv_lstm.summary()

    """
    Plot model into graph and save it into paths.MODEL_GRAPH
    """
    # plot_model(model_conv_lstm, to_file=paths.MODEL_GRAPH, show_shapes=True)

    return model_conv_lstm


def weighted_mse(y_true, y_pred):
    # size = 90
    # padding = 1
    y_pred_ = y_pred[::, 1:-1:, 1:-1:, 0]
    y_true_ = y_true[::, 1:-1:, 1:-1:, 0]
    #  weights = y_true[::, ::, ::, 1]

    mse = mean_squared_error(y_true_, y_pred_)
    unweighted_loss = K.square(y_true_ - y_pred_)
    # weighted_loss = weights * unweighted_loss
    # return K.in_train_phase(K.mean(unweighted_loss), K.mean(unweighted_loss))
    return K.in_train_phase(K.mean(unweighted_loss), K.mean(unweighted_loss))


def my_mean_squared_error(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse


def to_binary(img, trashold):
    return trashold < img


def get_mse_of_two_frames(first_frame, second_frame):
    """ Mean squared error """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    mse = tf.reduce_mean(tf.square(first_frame - second_frame))
    result = sess.run(mse)
    return result
