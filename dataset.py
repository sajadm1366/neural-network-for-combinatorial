import numpy as np
import tensorflow as tf


def data_samples(N, max_val=5, max_len=5):
    x = tf.convert_to_tensor(np.random.randint(max_val, size=(N, max_len)), dtype=tf.int32)
    y = tf.convert_to_tensor(np.sort(x), dtype=tf.int32)  # sort in ascending order
    return x, y


def data_gen(N, max_val=5, max_len=5):
    '''
    generate random integer numbers from 0-max_val with lenght of max_len
    inputes:
        N: numeber of sample
        max_len: max range
    outputs:
        x: unsorted inegers
        y: sorted x
    '''

    x, y = data_samples(N, max_val=max_val, max_len=max_len)

    return tf.data.Dataset.from_tensor_slices((x, y))
