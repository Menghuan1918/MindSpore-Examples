import argparse
import os
import random
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindvision.dataset import Mnist
from mindspore.dataset.vision import transforms

parser = argparse.ArgumentParser(description='MindSpore DCGAN Example')
parser.add_argument('--dataset', required=False,default='mnist',help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False,default='../data',help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use') #注意:多卡训练代码没写(写了我也没法验证)
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ",opt.manualSeed)
random.seed(opt.manualSeed)
mindspore.set_seed(opt.manualSeed)

if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

#暂时仅使用mnist数据集,其他暂时抛出异常
if opt.dataset == 'mnist':
    down_dataset = Mnist(path=opt.dataroot, split="train", batch_size=opt.batchSize, 
                    repeat_num=1, shuffle=True, resize=32, download=True)
    dataset = down_dataset.run()
else:
    raise ValueError("Just support mnist dataset now")
#使用transforms对数据集进行预处理
transforms_list = []
transforms_list.append(transforms.ToTensor())
transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
transform = transforms.Compose(transforms_list)


ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        mindspore.common.initializer.Normal()(m.weight)
    elif classname.find('BatchNorm') != -1:
        mindspore.common.initializer.Normal()(m.weight)
        mindspore.common.initializer.Zero()(m.bias)

# Generator Code [nnCell]
class Generator(nn.Cell):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.SequentialCell(
            nn.Conv2dTranspose(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0, pad_mode='valid'),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf, out_channels=1, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.Tanh()
        )

    def construct(self, input):
        return self.main(input)
    
# Create the Discriminator [nnCell]
class Discriminator(nn.Cell):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=ndf, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(num_features=ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, pad_mode='valid'),
            nn.Sigmoid()
        )

    def construct(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# Create the Generator
netG = Generator(ngpu)
netG.set_train(True)
# netG.apply(weights_init)
if opt.netG != '':
    netG.load_param(opt.netG)
print(netG)

# Create the Discriminator
netD = Discriminator(ngpu)
netD.set_train(True)
# netD.apply(weights_init)
if opt.netD != '':
    netD.load_param(opt.netD)
print(netD)

criterion = nn.BCELoss()

fixed_noise = ops.randn(opt.batchSize, nz, 1, 1)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = nn.Adam(netD.trainable_params(), learning_rate=opt.lr, beta1=opt.beta1)
optimizerG = nn.Adam(netG.trainable_params(), learning_rate=opt.lr, beta1=opt.beta1)

loss_monitor = LossMonitor(50)
trainD = Model(netD, optimizerD, criterion)
trainG = Model(netG, optimizerG, criterion)
#!!不能使用Model进行训练
# Training Loop
print("Starting Training Loop...")
for epoch in range(opt.niter):
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    # train with real
