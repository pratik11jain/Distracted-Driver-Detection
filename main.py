from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from keras import Model
from keras import backend as K
from keras.applications import xception, resnet50, mobilenet
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.cross_validation import LabelShuffleSplit
from sklearn.metrics import log_loss

if len(sys.argv) is not 4:
    print("Usage: python main.py <test_true/test_false> <downsample> <model_name>")
    sys.exit(1)

K.set_image_dim_ordering('tf')


def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def write_csv(predictions, ids, dest):
    df = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', pd.Series(ids, index=df.index))
    df.to_csv(dest, index=False)


# python main.py <test_true/test_false> <downsample> <model_name>

TESTING = True if sys.argv[1] == 'test_true' else False

DOWNSAMPLE = int(sys.argv[2])
data_path = 'dataset/data_{}'.format(DOWNSAMPLE)
DATASET_PATH = os.environ.get('DATASET_PATH', data_path + '.pkl' if not TESTING else data_path + '_subset.pkl')

DIR_PREFIX = 'DDD_MODEL'
MODEL_NAME = sys.argv[3]
FULL_DIR_PATH = DIR_PREFIX + '_' + MODEL_NAME + '_' + str(DOWNSAMPLE) + '/'
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', FULL_DIR_PATH + 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', FULL_DIR_PATH + 'summaries/')
MODEL_PATH = os.environ.get('MODEL_PATH', FULL_DIR_PATH + 'models/')

mkdirp(FULL_DIR_PATH)
mkdirp(CHECKPOINT_PATH)
mkdirp(SUMMARY_PATH)
mkdirp(MODEL_PATH)

NB_EPOCHS = 25 if not TESTING else 1
MAX_FOLDS = 3
NUM_CLASSES = 10

WIDTH, HEIGHT, NB_CHANNELS = 640 // DOWNSAMPLE, 480 // DOWNSAMPLE, 3
BATCH_SIZE = 50

with open(DATASET_PATH, 'rb') as f:
    X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)
    X_test = X_test.transpose(0, 2, 3, 1)
_, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)

predictions_total = []  # accumulated predictions from each fold
scores_total = []  # accumulated scores from each fold
accuracies_total = []  # accumulated accuracies from each fold
num_folds = 0


def choose_model(model_name):
    if model_name == 'vgg_bn':
        return vgg_bn()
    if model_name == 'vgg_16':
        return vgg_16()
    if model_name == 'vgg_19':
        return vgg_19()
    if model_name == 'my_model_2':
        return my_model_2()
    if model_name == 'xception':
        return xception_model()
    if model_name == 'le_net_5':
        return le_net_5()
    if model_name == 'res_net_50':
        return res_net_50()
    if model_name == 'alex_net':
        return alex_net()
    if model_name == 'mobile_net':
        return mobilenet_model()
    print('Incorrect model selection')
    sys.exit(1)


def vgg_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(WIDTH, HEIGHT, NB_CHANNELS)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def vgg_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(WIDTH, HEIGHT, NB_CHANNELS)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def alex_net(weights_path=None):
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=(4, 4), padding="same", kernel_initializer="he_normal",
                     data_format="channels_last",
                     input_shape=(WIDTH, HEIGHT, NB_CHANNELS)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (5, 5), activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, 3))

    model.add(Conv2D(384, (3, 3), activation="relu"))

    model.add(Conv2D(384, (3, 3), activation="relu"))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(3, 3))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def le_net_5(weights_path=None):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5),
                     activation='tanh',
                     input_shape=(WIDTH, HEIGHT, NB_CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def my_model(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=(WIDTH, HEIGHT, NB_CHANNELS)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def my_model_2(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=(WIDTH, HEIGHT, NB_CHANNELS)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def vgg_bn(weights_path=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", data_format="channels_last",
                     input_shape=(WIDTH, HEIGHT, NB_CHANNELS)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer="he_normal", activation="sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, kernel_initializer="he_normal", activation="softmax"))
    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def mobilenet_model():
    base_model = mobilenet.MobileNet(input_shape=(WIDTH, HEIGHT, NB_CHANNELS), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def xception_model():
    base_model = xception.Xception(input_shape=(WIDTH, HEIGHT, NB_CHANNELS), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def res_net_50():
    base_model = resnet50.ResNet50(input_shape=(WIDTH, HEIGHT, NB_CHANNELS), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


for train_index, valid_index in LabelShuffleSplit(driver_indices, n_iter=MAX_FOLDS, test_size=0.2, random_state=67):
    print('Fold {}/{}'.format(num_folds + 1, MAX_FOLDS))

    X_train, y_train = X_train_raw[train_index, ...], y_train_raw[train_index, ...]
    X_valid, y_valid = X_train_raw[valid_index, ...], y_train_raw[valid_index, ...]
    X_train = X_train.transpose(0, 2, 3, 1)
    X_valid = X_valid.transpose(0, 2, 3, 1)
    model = choose_model(MODEL_NAME)
    model_path = os.path.join(MODEL_PATH, 'model_{}.json'.format(num_folds))
    with open(model_path, 'w') as f:
        f.write(model.to_json())

    # restore existing checkpoint, if it exists
    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds))
    if os.path.exists(checkpoint_path):
        print('Restoring fold from checkpoint.')
        model.load_weights(checkpoint_path)

    summary_path = os.path.join(SUMMARY_PATH, 'model_{}'.format(num_folds))
    mkdirp(summary_path)

    # Model fit params
    MODEL_MONITOR = 'val_acc'
    MODEL_PATIENCE = 5

    callbacks = [
        EarlyStopping(monitor=MODEL_MONITOR, patience=MODEL_PATIENCE, verbose=1, mode='auto'),
        ModelCheckpoint(checkpoint_path, monitor=MODEL_MONITOR, verbose=1, save_best_only=True, mode='auto'),
        # TensorBoard(log_dir=summary_path, histogram_freq=1, write_graph=True, write_images=True)  # To Do
        TensorBoard(log_dir=summary_path, write_graph=True, write_images=True)  # To Do
    ]
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE, epochs=NB_EPOCHS,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks)

    predictions_valid = model.predict(X_valid, batch_size=100, verbose=1)

    score_valid = log_loss(y_valid, predictions_valid)
    scores_total.append(score_valid)
    print('Score: {}'.format(score_valid))

    predictions_test = model.predict(X_test, batch_size=100, verbose=1)
    predictions_total.append(predictions_test)

    num_folds += 1

min_value = min(scores_total)
min_val, min_index = min((scores_total[i], i) for i in xrange(len(scores_total)))

print('Final chosen model is {} with loss: {}'.format(min_index + 1, scores_total[min_index]))

write_csv(predictions_total[min_index], X_test_ids,
          os.path.join(SUMMARY_PATH, 'predictions_{}_{:.2}.csv'.format(int(time.time()), scores_total[min_index])))



# Code built on top of: https://github.com/fomorians/distracted-drivers-keras
