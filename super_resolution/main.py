import argparse

import mindspore as ms
import mindspore.nn as nn
from mindspore import Model
from mindspore.dataset.vision.py_transforms import Compose
from mindspore.train.callback import LossMonitor
import mindspore.dataset as data
from mindspore.dataset.vision import transforms

import urllib.request
from os.path import exists, join, basename
from os import makedirs, remove,listdir
import tarfile
from PIL import Image
from model import Net
#dataset---------------------------
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class DatasetFromFolder():
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = input.map(operations=self.input_transform, input_columns="image")
        if self.target_transform:
            target = target.map(operations=self.target_transform, input_columns="image")

        return input, target

    def __len__(self):
        return len(self.image_filenames)

#data---------------------------
def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    #使用transforms对数据集进行预处理
    transform_input = [
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size // upscale_factor),
        transforms.ToTensor()
    ]
    transform_target = [
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ]
    return DatasetFromFolder(train_dir,
                            input_transform=transform_input,
                            target_transform=transform_target)

def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    #使用transforms对数据集进行预处理
    transform_input = [
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size // upscale_factor),
        transforms.ToTensor()
    ]
    transform_target = [
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ]
    return DatasetFromFolder(test_dir,
                             input_transform=transform_input,
                             target_transform=transform_target)

# Training settings
parser = argparse.ArgumentParser(description='MindSpore Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3 , help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--no-gpu', action='store_true', default=False,help='disables GPU training')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

if opt.no_gpu:
    ms.set_context(device_target="CPU")
ms.set_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data = data.GeneratorDataset(train_set,column_names=["data"],
                                      num_parallel_workers = opt.threads,shuffle=True)
training_data = training_data.batch(opt.batchSize, drop_remainder=True)
testing_data = data.GeneratorDataset(test_set,column_names=["data"], 
                                     num_parallel_workers = opt.threads,shuffle=False)
testing_data = testing_data.batch(opt.testBatchSize, drop_remainder=True)
print('===> Building model')

model = Net(upscale_factor=opt.upscale_factor)
criterion = nn.MSELoss(reduction='mean')
optimizer = nn.Adam(model.trainable_params(), learning_rate=opt.lr)

train_model = Model(network=model, loss_fn=criterion, optimizer=optimizer, metrics=None)

# Training
for epoch in range(1, opt.nEpochs + 1):
    train_model.train(epoch = epoch,train_dataset = training_data,callbacks=[LossMonitor()])
    train_model.eval(testing_data,callbacks=[LossMonitor()])
    #save_checkpoint(model, epoch)
    ms.save_checkpoint(model, "checkpoint_{}.ckpt".format(epoch))