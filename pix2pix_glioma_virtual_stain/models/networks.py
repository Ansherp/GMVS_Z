import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np


from models.HRnet import get_pose_HRnet
from models.dinknet import DinkNet34
from models.CrossModel.CrossWnet import Cross_UNet
from models.transunet import TransUNet


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[], conv_nd=2):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, conv_nd)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'UNet':
        netG = GeneratorUNet(input_nc, output_nc, conv_nd)
    elif netG == 'trans_UNet':
        netG = TransUNet(input_nc, output_nc)

    #  Cross Net
    elif netG == 'Cross':
        netG = Cross_UNet(input_nc)

    #  D-Link Net
    elif netG == 'DinkNet34':
        netG = DinkNet34(input_nc, output_nc)

    # HR net
    elif netG == "HRnet":
        netG = get_pose_HRnet()

    else:
        raise ('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda() if len(gpu_ids) != 0 else Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####           
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model

        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample            
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]

            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, conv_nd=2,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        self.conv_nd = conv_nd
        activation = nn.ReLU(True)

        if conv_nd == 2:
            model1 = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, padding_mode="reflect"), norm_layer(ngf),
                      activation]
        else:
            model1 = [nn.Conv3d(1, ngf, kernel_size=5, stride=1, padding=2), nn.BatchNorm3d(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if conv_nd == 2:
                model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult * 2), activation]
            else:
                model1 += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=(4, 1, 1)),
                           nn.BatchNorm3d(ngf * mult * 2), activation]
        if conv_nd == 3:
            model1 += [nn.Conv3d(ngf * mult * 2, ngf * mult * 2, kernel_size=7, stride=1, padding=(0, 3, 3)),
                       nn.BatchNorm3d(ngf * mult * 2), activation]
        ### resnet blocks
        mult = 2 ** n_downsampling
        model2 = []
        for i in range(n_blocks):
            model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model2 += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, input):
        if self.conv_nd == 3:
            input = input.unsqueeze(0)
        down = self.model1(input)
        if self.conv_nd == 3:
            down = down.squeeze(2)
        out = self.model2(down)

        return out


##############################
#           U-NET
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, conv_nd=2, normalize=True, dropout=0.0, pad3d=1):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False) if conv_nd == 2 else nn.Conv3d(in_size, out_size,
                                                                                             5, 2,
                                                                                             (pad3d, 1, 1),
                                                                                             bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size) if conv_nd == 2 else nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout) if conv_nd == 2 else nn.Dropout3d(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, n):
        super(ResnetBlock, self).__init__()
        conv_block = []
        for i in range(n):
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.InstanceNorm2d(dim), nn.ReLU()]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, conv_nd=2, dropout=0.0, pad3d=1):
        super(UNetUp, self).__init__()
        # conv_nd = 2  # 2D3D卷积转换

        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False) if conv_nd == 2 else nn.ConvTranspose3d(in_size,
                                                                                                               out_size,
                                                                                                               5, 2,
                                                                                                               (
                                                                                                                   pad3d,
                                                                                                                   1,
                                                                                                                   1),
                                                                                                               bias=False),
            nn.InstanceNorm2d(out_size) if conv_nd == 2 else nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]

        if dropout:
            layers.append(nn.Dropout(dropout) if conv_nd == 2 else nn.Dropout3d(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, conv_nd=2):
        super(GeneratorUNet, self).__init__()
        self.conv_nd = conv_nd

        self.down1 = UNetDown(in_channels if conv_nd == 2 else 1, 64, conv_nd, normalize=False)  # 3d_C = 6
        self.down2 = UNetDown(64, 128, conv_nd)  # 3d_C = 6
        self.down3 = UNetDown(128, 256, conv_nd)  # 3d_C = 6
        self.down4 = UNetDown(256, 512, conv_nd)  # 3d_C = 6
        self.down5 = UNetDown(512, 512, conv_nd)

        self.up1 = UNetUp(512, 512, conv_nd)
        self.up2 = UNetUp(1024, 256, conv_nd)
        self.up3 = UNetUp(512, 128, conv_nd)
        self.up4 = UNetUp(256, 64, conv_nd)

        self.con3d22d = nn.Sequential(nn.Conv3d(128, 1, 5, 1, 2),
                                      nn.InstanceNorm3d(1),
                                      nn.ReLU(),
                                      nn.ReplicationPad3d((1, 0, 1, 0, 0, 0)))  # 通道数由128变换为32

        if conv_nd == 2:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(128, out_channels, 4, padding=1),
                nn.Tanh(),
            )
        else:
            self.final = nn.Sequential(
                nn.ConvTranspose2d(31, out_channels, 4, 2, padding=1),
                nn.Tanh(),
            )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        if self.conv_nd == 3:
            x = x.unsqueeze(1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # u3 = self.up3(d4, d5)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        if self.conv_nd == 3:
            # print("###############u4", u4.shape)
            u4 = self.con3d22d(u4)
            # print("###############conv3d to 2d u4", u4.shape)
            u4 = u4.squeeze(1)
        out = self.final(u4)
        # print("###############out", out.shape)
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


##############################
#           Trans_U-NET
##############################
class Trans_UNetDown(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Trans_UNetDown, self).__init__()
        self.model1 = nn.Sequential(nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
                                    nn.InstanceNorm2d(out_size),
                                    nn.LeakyReLU(0.2), )
        layers = [
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model1(x)
        return self.model(x)


class Trans_UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Trans_UNetUp, self).__init__()

        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Trans_UNet3D(nn.Module):
    def __init__(self, in_size, out_size):
        super(Trans_UNet3D, self).__init__()

        layers = [
            nn.Conv3d(in_size, out_size, 3, 1, 1),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_size, out_size, 5, 1, 2),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_size, out_size, 7, 1, 3),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class Trans_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, conv_nd=2):
        super(Trans_UNet, self).__init__()
        self.conv_nd = conv_nd
        if conv_nd == 2:
            self.init_layer = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1),
                                            nn.InstanceNorm2d(32),
                                            nn.LeakyReLU(0.2),
                                            )
        else:
            self.init_layer = Trans_UNet3D(1, 1)
            self.layer_3d22d = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1),
                                             nn.InstanceNorm2d(32),
                                             nn.LeakyReLU(0.2),
                                             )
        self.down1 = Trans_UNetDown(32, 64)  # 3d_C = 6
        self.down2 = Trans_UNetDown(64, 128)  # 3d_C = 6
        self.down3 = Trans_UNetDown(128, 256)  # 3d_C = 6
        self.down4 = Trans_UNetDown(256, 512)  # 3d_C = 6
        self.down5 = Trans_UNetDown(512, 512)

        self.up1 = Trans_UNetUp(512, 512)
        self.up2 = Trans_UNetUp(1024, 256)
        self.up3 = Trans_UNetUp(512, 128)
        self.up4 = Trans_UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # print("$$$$$$$$$$$$$$$$$$$$", x.shape)
        # U-Net generator with skip connections from encoder to decoder
        if self.conv_nd == 2:
            d0 = self.init_layer(x)
        else:
            x = x.unsqueeze(1)
            # print("#############x###############", x.shape)
            d = self.init_layer(x)
            # print("#############d###############", d.shape)
            d0 = self.layer_3d22d(d.squeeze(1))
            # print("#############d0###############", d0.shape)

        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # u3 = self.up3(d4, d5)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)


if __name__ == "__main__":
    model_global = GlobalGenerator(3, 3, ngf=32, n_downsampling=4, n_blocks=9, norm_layer=nn.BatchNorm2d, conv_nd=2,
                                   padding_type='reflect')
    model_global.load_state_dict(
        torch.load(r"C:\Users\Hai\Desktop\latest_net_G.pth"), strict=False)
    print('gobal net load')
    model_global = model_global.model
    pass
