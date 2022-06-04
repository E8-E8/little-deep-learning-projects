import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# create features

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

#  creaete labels

y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# visualize the data

plt.scatter(X, y)
# plt.show()

houseInfo = tf.constant(['bedroom', 'bathroom', 'garage'])
housePrice = tf.constant([939700])


X = tf.constant(X)
y = tf.constant(y)


# creating a model
