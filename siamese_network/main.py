import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        