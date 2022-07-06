import numpy as np
import tensorflow as tf
from dataset import data_gen


def test_data_gen():
    data = data_gen(N=5, max_val=5, max_len=5)
    x, y = next(iter(data.take(1)))
    assert x.shape == (5, )
    assert y.shape == (5, )
    for x, y in data:
        assert all(np.sort(x) == y.numpy())



test_data_gen()


data = data_gen(N=1, max_val=5, max_len=5)


for b in range(2):
    for x, _ in data:
        print(x)