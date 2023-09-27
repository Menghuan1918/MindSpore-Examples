import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import matplotlib.pyplot as plt
from mindspore.train import Model
from mindvision.engine.callback import LossMonitor
import mindspore.dataset.transforms as trans

def data_generate():
    np.random.seed(2)
    T = 20
    L = 1000
    N = 100
    x = np.empty((N, L), "int64")
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype(np.float32)
    return data

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(1, 51)
        self.lstm2 = nn.LSTM(51, 51)
        self.linear = nn.Dense(51, 1)

    def construct(self, input, future=0):
        outputs = []
        h_t = ops.zeros((1, 1, 51), dtype=ms.float32)
        c_t = ops.zeros((1, 1, 51), dtype=ms.float32)
        h_t2 = ops.zeros((1, 1, 51), dtype=ms.float32)
        c_t2 = ops.zeros((1, 1, 51), dtype=ms.float32)

        for input_t in ops.split(ms.Tensor(input), 1):
            input_t = input_t.reshape((1, -1, 1))  # Reshape input_t
            output, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))  # Unpack the outputs
            output, (h_t2, c_t2) = self.lstm2(output, (h_t2, c_t2))  # Unpack the outputs
            output = self.linear(output)
            outputs += [output]

        for i in range(future):
            input_t = input_t.reshape((1, -1, 1))  # Reshape input_t
            output, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))  # Unpack the outputs
            output, (h_t2, c_t2) = self.lstm2(output, (h_t2, c_t2))  # Unpack the outputs
            output = self.linear(output)
            outputs += [output]

        outputs = ms.ops.Concat(1)(outputs)
        return outputs


def Dataconstruct():
    data = data_generate()
    get_data = data[3:, :-1].astype(np.float32)
    label = data[3:, 1:].astype(np.float32)
    dataset = ds.NumpySlicesDataset({"data": get_data, "label": label}, shuffle=False)
    return dataset

def Dataconstruct_test():
    data = data_generate()
    get_data = data[:3, :-1].astype(np.float32)
    dataset = ds.NumpySlicesDataset({"data": get_data}, shuffle=False)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    ms.set_seed(0)
    #load the data
    data = data_generate()
    # build the model
    seq = Net()
    # loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = nn.SGD(seq.trainable_params(), learning_rate=0.8)
    model = Model(network=seq, loss_fn=loss_fn, optimizer=optimizer)
    train_data = Dataconstruct()

from mindspore import load_checkpoint, load_param_into_net
for i in range(opt.steps):
    print('step: ', i)
    model.train(epoch = 1, train_dataset = train_data,callbacks=[LossMonitor(0.01, 1)])
# 保存模型
print("Saving.....")
save_checkpoint_path = "lstm_model.ckpt"
ms.save_checkpoint(seq, save_checkpoint_path)
# 加载模型
print("Loading.....")
loaded_seq = Net()
param_dict = load_checkpoint(save_checkpoint_path)
param_not_load = load_param_into_net(loaded_seq, param_dict)
loaded_model = Model(network=loaded_seq, loss_fn=loss_fn, optimizer=optimizer)

predictions = []
last_train_input = Dataconstruct_test()
print("Start")

for data in last_train_input.create_dict_iterator():
    input_tensor = data["data"]
    predicted = loaded_model.predict(input_tensor)
    predictions.append(predicted.asnumpy())
predictions = np.array(predictions)
print(predictions.shape)
print("Start plt")
# 将预测结果绘制出来->>>待添加原始数据
future = 1000
plt.figure(figsize=(12, 6))
plt.title("Predictions for Next 1000 Points")

def draw(yi, color):
    plt.plot(np.arange(yi.shape[1]), yi[0, :, 0], color, linewidth=2.0)
    #plt.plot(np.arange(yi.shape[1], yi.shape[1] + future), yi[0, yi.shape[1]:, 0], color + ':', linewidth=2.0)

draw(predictions[0, :, :, :], 'r')
draw(predictions[1, :, :, :], 'g')
draw(predictions[2, :, :, :], 'b')

plt.legend(['Prediction 1', 'Prediction 1 Future', 'Prediction 2', 'Prediction 2 Future', 'Prediction 3', 'Prediction 3 Future'])
plt.savefig('predictions.png')
plt.show()