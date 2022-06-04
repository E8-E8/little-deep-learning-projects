import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# labels
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0,
             11.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0,
             21.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])

# transfor arrays into tensors
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)

tf.random.set_seed(42)

# create a model using the Sequencial API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_shape=[1], activation=None),
    # activation relu is used to avoid overfitting
    tf.keras.layers.Dense(1, input_shape=[1]),
])

# compile the model
model.compile(loss=tf.keras.losses.mae,  # mea - mean absolute error
              # sgd - stochastic gradient descent
              # adam - adaptive momentum is better than sgd
              # lr - learning rate (how fast the model learns, smaller the better)
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mae'])
# fit the model
model.fit(X, y, epochs=100)
# batch size - number of samples to use in each iteration
# check out x and y
print(X, y)

# make a model prediction
print(model.predict([100.0]))


# improve model
# the learning rate is most important hyperparameter in the model

# evaluating the model
