import numpy as np
import tensorflow as tf
import os
import tensorflow_probability as tfp

scalar = tf.constant(7)

# create a vector of size 3
vector = tf.constant([10, 10])

# check the dimension of the vector

# cretate a matrix of size 2x3
matrix = tf.constant([[1, 2], [4, 5]])

matrix.ndim

# create another matrix of size 3x2
matrix2 = tf.constant([[6., 7.], [8., 9.], [10., 11.]], dtype=tf.float16)

changebleTensor = tf.Variable([10, 7])
unchangableTensor = tf.constant([10, 7])

# create two random tensors
randomTensor = tf.random.Generator.from_seed(42)
randomTensor = randomTensor.normal(shape=(3, 2))
randomTensor2 = tf.random.Generator.from_seed(42)
randomTensor2 = randomTensor2.normal(shape=(3, 2))

tf.random.set_seed(56)
not_shuffled = tf.constant([[1, 2], [3, 4], [5, 6]])
shuffled = tf.random.shuffle(not_shuffled, seed=42)

# create a rank 2 tensor
tensor2 = tf.constant([[1, 2, 5], [3, 4, 9], ])
# create a rank 3 tensor
tensor3 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# add another extra dimension to the tensor
tensor4 = tensor2[..., tf.newaxis]


tensor = tf.constant([[10, 7], [8, 9]])
tensor = tensor + 10  # add 10 to each element of the tensor

# tf build in functions

# print(tf.multiply(tensor, 3))  # better to use his one becouse it uses gpu
# print(tf.add(tensor, 5))

# matrix multiplication


tensor1 = tf.constant([[1, 2, 5], [7, 2, 1], [3, 3, 3]])
tensor2 = tf.constant([[3, 5], [6, 7], [1, 8]])


print(tensor1.shape)
print(tensor2.shape)
print(tf.matmul(tensor1, tensor2))

# reshape a tensor

tensorNew = tf.reshape(tensor2, shape=(2, 3))

print(tensorNew, tensorNew.ndim)

# changing the data type of a tensor
B = tf.constant([1., 2., 3., 4., 5.])
print(B.dtype)

C = tf.constant([3, 4])
print(C.dtype)

print(tf.__version__)

B = tf.cast(B, tf.float16)

# change fro int32 to float32
C = tf.cast(C, tf.float32)

# Aggregation of tensors

# getting the absolute value of a tensor
tensor = tf.constant([[1., 2.], [3., 4.]])
print(tensor)
print(tf.abs(tensor))

# get the minimum value of a tensor
print(tf.reduce_min(tensor))
# get the maximum value of a tensor
print(tf.reduce_max(tensor))
# get the mean value of a tensor
print(tf.reduce_mean(tensor))
# get the sum of a tensor
print(tf.reduce_sum(tensor))
# get the variance of a tensor
print(tf.math.reduce_variance(tensor))
# get the standard deviation of a tensor
print(tf.math.reduce_std(tensor))
# get the positional minimum of a tensor
print(tf.math.argmin(tensor))
# get the positional maximum of a tensor
print(tf.math.argmax(tensor))

tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
tf.cast(F, tf.float32)
print(tf.math.argmax(F))
print(tf.math.argmin(F))
print(tf.math.reduce_max(F))
print(tf.math.reduce_min(F))

# squeezeing a tensor
tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
print(G)
GSqueezed = tf.squeeze(G)  # removes the extra dimensions
print(GSqueezed)

# create a list of indices
some_list = [0, 1, 2, 3]  # could be red, green, blue, purple

# one hot encode our list
one_hot_encoded = tf.one_hot(some_list, depth=4)
print(one_hot_encoded)

# specify custom values for the one hot encoded tensor
one_hot_encoded = tf.one_hot(
    some_list, depth=4, on_value="hello", off_value="world")

print(one_hot_encoded)

# squaring log squre root
H = tf.range(1, 11)
H = tf.cast(H, tf.float64)
print(H)
# methods require not int types
print(tf.math.square(H))
print(tf.math.sqrt(H))
print(tf.math.log(H))

I = tf.range(1, 11)
J = tf.range(1, 11)
I = tf.cast(I, dtype=tf.float32)


# print(tf.math.equal(I, J))
print(tf.math.exp(I))
print(tf.math.floor(I))

# tensors an numpy

K = tf.constant(np.array([[1, 2, 3], [4, 5, 6]]))

# conver tensor to numpy array

print(np.array(K))

# default types of each are slightly different
numpy_J = tf.constant(np.array([[1., 2., 3.], [4., 5., 6.]]))
tensor_J = tf.constant([[1., 2., 3.], [4., 5., 6.]])

print(numpy_J.dtype)
print(tensor_J.dtype)

print(tf.test.is_gpu_available())
print(tf.test.benchmark_config())
