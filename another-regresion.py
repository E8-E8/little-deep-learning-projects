from tensorflow.keras.utils import plot_model

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np


X = tf.cast(tf.range(-100, 100, 4), dtype=tf.float32)

y = X + 10


X_train = X[:40]  # 80% of the data

y_train = y[:40]


X_test = X[40:]  # 20% of the data

y_test = y[40:]


tf.random.set_seed(42)


model = tf.keras.Sequential([

    tf.keras.layers.Dense(

        50, input_shape=[1], name="input", activation=None),

    tf.keras.layers.Dense(1, input_shape=[1], name="output")

], name="regression")


model.compile(loss=tf.keras.losses.mae,

              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),

              metrics=['mae'])


model.fit(X_train, y_train, epochs=500, verbose=0)


y_pred = model.predict(X_test)


def plot_predictions(train_data=X_train,

                     train_labels=y_train,

                     test_data=X_test, test_labels=y_test, predictions=y_pred):

    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c='b', label='Training data')

    plt.scatter(test_data, test_labels, c='g', label='Testing data')
    # plot model's predictions in red

    plt.scatter(test_data, predictions, c='r', label='Predictions')

    plt.legend()

    plt.show()


def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true, y_pred)


plot_predictions()

# Experiments to improve the model
