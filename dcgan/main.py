import argparse
import os
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
    down_dataset = Mnist(path=opt.dataroot, split="train", batch_size=opt.batchSize, 
                    repeat_num=1, shuffle=True, resize=32, download=True)
    dataset = down_dataset.run()
    nc = 1
else:
    raise ValueError("Just support mnist dataset now")
#使用transforms对数据集进行预处理
transform = [
    transforms.Resize(opt.imageSize),
    lambda x: (x[0],),
    #transforms.HWC2CHW(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
]
dataset = dataset.map(operations=transform,input_columns="image").batch(opt.batchSize, drop_remainder=True)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

# custom weights initialization called on net functions
# Generator Code [nnCell]
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
def g_forward(_z, _valid):
    _gen_imgs = netG(_z)
    _g_loss = loss_function(netD(_gen_imgs), _valid)
    return _g_loss, _gen_imgs

def d_forward(_real_imgs, _gen_imgs, _valid, _fake):
    real_loss = loss_function(netD(_real_imgs), _valid)
    fake_loss = loss_function(netD(_gen_imgs), _fake)
    _d_loss = (real_loss + fake_loss) / 2
    return _d_loss

grad_g = ops.value_and_grad(g_forward, None, netG.trainable_params(),has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, netD.trainable_params(),has_aux=False)

if opt.dry_run:
    opt.niter = 1
# Training Loop
print("Starting Training Loop...")
for epoch in range(opt.niter):
    print(epoch)
    netD.set_train()
    netG.set_train()
    for i, (data,_) in enumerate(dataset):
        # train with real
        optimizerD.clear_grad()
        real_cpu = data[0]
        batch_size = real_cpu.shape[0]
        label = ops.ones((batch_size, 1, 1, 1), mindspore.float32)
        output = netD(real_cpu)
        errD_real = loss_function(output, label)
        errD_real.backward()
        D_x = output.mean().asnumpy()
        # train with fake
        noise = ops.randn(batch_size, nz, 1, 1)
        fake = netG(noise)
        label = ops.zeros((batch_size, 1, 1, 1), mindspore.float32)
        output = netD(fake.detach())
        errD_fake = loss_function(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().asnumpy()
        errD = errD_real + errD_fake
        optimizerD.step()
        # (2) Update G network: maximize log(D(G(z)))
        optimizerG.clear_grad()
        label = ops.ones((batch_size, 1, 1, 1), mindspore.float32)
        output = netD(fake)
        errG = loss_function(output, label)
        errG.backward()
        D_G_z2 = output.mean().asnumpy()