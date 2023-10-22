import argparse
import os
import gzip
import shutil
import urllib.request
import random
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindvision.dataset import Mnist
import mindspore.common.initializer as init
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
    file_path = "../../data/MNIST/"
    nc = 1
    if not os.path.exists(file_path):
        # 下载数据集
        if not os.path.exists('../../data'):
            os.mkdir('../../data')
        os.mkdir(file_path)
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                    't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        for file_name in file_names:
            url = (base_url + file_name).format(**locals())
            print("正在从" + url + "下载MNIST数据集...")
            urllib.request.urlretrieve(url, os.path.join(file_path, file_name))
            with gzip.open(os.path.join(file_path, file_name), 'rb') as f_in:
                print("正在解压数据集...")
                with open(os.path.join(file_path, file_name)[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(os.path.join(file_path, file_name))
#使用transforms对数据集进行预处理
transform = [
    transforms.Resize(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
]
dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=transform, input_columns="image").batch(opt.batchSize)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

# custom weights initialization called on net functions
class Generator(nn.Cell):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.SequentialCell(
            nn.Conv2dTranspose(in_channels=nz, out_channels=ngf * 8, kernel_size=4,
                                stride=1, padding=0, pad_mode='valid',
                                weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(num_features=ngf * 8,
                            gamma_init=init.Normal(0.02, 1.0),
                            beta_init=init.Constant(0.0)),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4,
                                stride=2, padding=1, pad_mode='pad',
                                weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(num_features=ngf * 4,
                            gamma_init=init.Normal(0.02, 1.0),
                            beta_init=init.Constant(0.0)),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4,
                                stride=2, padding=1, pad_mode='pad',
                                weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(num_features=ngf * 2,
                            gamma_init=init.Normal(0.02, 1.0),
                            beta_init=init.Constant(0.0)),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, 
                               stride=2, padding=1, pad_mode='pad',
                               weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(num_features=ngf,
                            gamma_init=init.Normal(0.02, 1.0),
                            beta_init=init.Constant(0.0)),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels=ngf, out_channels=nc, kernel_size=4,
                                stride=2, padding=1, pad_mode='pad',
                                weight_init=init.Normal(0.02, 0.0)),
            nn.Tanh()
        )
    def construct(self, input):
        print(f"Ginput shape:{input.shape}")
        output = self.main(input)
        return output
    
# Create the Discriminator [nnCell]
class Discriminator(nn.Cell):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.SequentialCell(
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, 
                      stride=2, padding=1, pad_mode='pad',
                      weight_init=init.Normal(0.02, 0.0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4,
                       stride=2, padding=1, pad_mode='pad',
                       weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(num_features=ndf * 2,
                            gamma_init=init.Normal(0.02, 1.0),
                            beta_init=init.Constant(0.0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4,
                       stride=2, padding=1, pad_mode='pad',
                       weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(num_features=ndf * 4,
                            gamma_init=init.Normal(0.02, 1.0),
                            beta_init=init.Constant(0.0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4,
                       stride=2, padding=1, pad_mode='pad',
                       weight_init=init.Normal(0.02, 0.0)),
            nn.BatchNorm2d(num_features=ndf * 8,
                            gamma_init=init.Normal(0.02, 1.0),
                            beta_init=init.Constant(0.0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4,
                       stride=1, padding=0, pad_mode='valid',
                       weight_init=init.Normal(0.02, 0.0)),
            nn.Sigmoid()
        )
    def construct(self, input):
        print(f"Dinput shape:{input.shape}")
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# Create the Generator and Discriminator
netG = Generator(ngpu)
if opt.netG != '':
    netG.load_param(opt.netG)
netD = Discriminator(ngpu)
if opt.netD != '':
    netD.load_param(opt.netD)

loss_function = nn.BCELoss()

# setup optimizer
optimizerD = nn.Adam(netD.trainable_params(), learning_rate=opt.lr, 
                     beta1=opt.beta1, beta2=0.999)
optimizerG = nn.Adam(netG.trainable_params(), learning_rate=opt.lr, 
                     beta1=opt.beta1, beta2=0.999)

fixed_noise = ops.randn(opt.batchSize, nz, 1, 1)
real_label = 1
fake_label = 0

#Generator forward function
def forward(_z, _valid):
    _g_loss = loss_function(_z, _valid)
    return _g_loss

grad_d = ops.value_and_grad(forward, None, optimizerD.parameters,has_aux=False)
grad_g = ops.value_and_grad(forward, None, optimizerG.parameters,has_aux=True)

if opt.dry_run:
    opt.niter = 1
# Training Loop
print("Starting Training Loop...")
for epoch in range(opt.niter):
    print(epoch)
    netD.set_train()
    netG.set_train()
    for i, (data,a) in enumerate(dataset.create_tuple_iterator()):
        # train with real
        real_cpu = data
        batch_size = real_cpu.shape[0]
        label = ops.full((batch_size,), real_label, dtype=real_cpu.dtype)
        print("Start training Discriminator...")
        output = netD(real_cpu)
        print(f"output: {output.shape} label:{label.shape}")
        (errD_real,_),grad_errD_real = grad_d(output, label)#errD_real = criterion(output, label)
        optimizerD(grad_errD_real)#errD_real.backward()
        D_x = output.mean().asnumpy()

        print("Start training Generator...")
        # train with fake
        noise = ops.randn(batch_size, nz, 1, 1)
        fake = netG(noise)
        label.fill(fake_label)
        output = netD(fake.detach())
        (errD_fake,_),grad_errD_fake = grad_g(output, label)
        optimizerD(grad_errD_fake)
        D_G_z1 = output.mean().asnumpy()
        errD = errD_real + errD_fake
        
        label.fill(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        (errG,_),grad_errG = grad_g(output, label)
        optimizerG(grad_errG)
        D_G_z2 = output.mean().asnumpy()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataset),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))