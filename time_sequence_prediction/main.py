import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import matplotlib.pyplot as plt
from mindspore.train import Model,LearningRateScheduler
from mindvision.engine.callback import LossMonitor

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
        print(input.shape)
        a = input.shape[0]
        h_t = ops.zeros((a, 51),dtype=ms.float32)
        c_t = ops.zeros((a, 51),dtype=ms.float32)
        h_t2 = ops.zeros((a, 51),dtype=ms.float32)
        c_t2 = ops.zeros((a, 51),dtype=ms.float32)
        #output = ops.zeros((2, 2), dtype=ms.float32).asnumpy()
        print(input.shape)
        for input_t in ops.split(ms.Tensor(input),1):
            print(input_t.shape)
            aaa = self.lstm1(input_t)
            #aaa = self.lstm1(input_t, (h_t, c_t))
            print(aaa.shape)
            h_t,c_t = aaa
            aaa2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t2, c_t2 = aaa2
            output = self.linear(h_t2)
            outputs += [output]
        
        for i in range(future):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = ms.ops.Concat(1)(outputs)
        return outputs

def Dataconstruct():
    data = data_generate()
    get_data = data[3:, :-1].astype(np.float32)
    label = data[3:, 1:].astype(np.float32)
    return get_data, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    ms.set_seed(0)
    #load the data
    data = data_generate()
    input = data[3:, :-1]
    target = data[3:, 1:]
    test_input = ms.Tensor(data[:3, :-1], ms.float32)
    test_target = ms.Tensor(data[:3, 1:], ms.float32)
    # build the model
    seq = Net()
    # loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = nn.SGD(seq.trainable_params(), learning_rate=0.8)
    model = Model(network=seq, loss_fn=loss_fn, optimizer=optimizer)
    loader = Dataconstruct()
    train_data = ds.GeneratorDataset(source=loader,column_names=["data", "label"])
    # train the model
    for i in range(opt.steps):
        print('step: ', i)
        model.train(epoch = 1, train_dataset = train_data,callbacks=[LossMonitor(0.01, 10)])
        # begin to predict !!!!!!!!!!!!!!!!!!!!!!!!!
        print("!!After train!!")
        future = 1000
        pred = seq(test_input, future=future)
        loss = loss_fn(pred[:, :-future], test_target)
        print("test loss is {}".format(loss))
        y = pred.asnumpy()

        
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()