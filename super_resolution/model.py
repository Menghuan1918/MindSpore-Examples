import mindspore.nn as nn
import mindspore.common.initializer as init
import numpy as np

class CustomInit(init.Initializer):
    def __init__(self, gain=1.0):
        super(CustomInit, self).__init__()
        self.gain = gain

    def _initialize(self, arr):
        # 正交初始化代码
        flat_shape = (arr.size, ) if len(arr.shape) < 2 else (arr.shape[0], np.prod(arr.shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(arr.shape)
        return (self.gain * q[:arr.size]).astype(arr.dtype)

class Net(nn.Cell):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, pad_mode='pad', weight_init=CustomInit(np.sqrt(2)))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init=CustomInit(np.sqrt(2)))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init=CustomInit(np.sqrt(2)))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, stride=1, padding=1, pad_mode='pad', weight_init=CustomInit())
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

