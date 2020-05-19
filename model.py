import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import get_loader

class GatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super(GatedConv2d, self).__init__()
        self.c1 = nn.Conv2d(in_channel, out_channel, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding, 
                            bias=bias)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding, 
                            bias=bias)
        self.n2 = nn.InstanceNorm2d(out_channel)
    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        
        x2 = self.c2(x)
        x2 = self.n2(x2)
        x3 = x1 * torch.sigmoid(x2)
        return x3

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            GatedConv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            GatedConv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_speakers=6, repeat_num=6):
        super(Generator, self).__init__()
        c_dim = num_speakers
        layers = []
        layers.append(GatedConv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4)))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(GatedConv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3)))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(GatedConv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5, num_speakers=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(GatedConv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(GatedConv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = GatedConv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0) # padding should be 0
        self.conv_clf_spks = GatedConv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0)  # for num_speaker
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))

class GatedConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(GatedConv1d, self).__init__()
        self.c1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm1d(out_channel)
        self.c2 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm1d(out_channel)
        
    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x2 = self.c2(x)
        x2 = self.n2(x2)
        x3 = x1 * torch.sigmoid(x2)
        return x3

class ResidualBlock1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1d, self).__init__()
        self.main = nn.Sequential(
            GatedConv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            GatedConv1d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True))
    
    def forward(self, x):
        return x + self.main(x)
    
class Generatorf0(nn.Module):
    def __init__(self, conv_dim=64, num_speakers=6, repeat_num=6, scale=10):
        super(Generatorf0, self).__init__()
        c_dim = num_speakers
        layers = []
        layers.append(GatedConv1d(scale+c_dim, conv_dim, kernel_size=15, stride=1, padding=7))
        layers.append(nn.InstanceNorm1d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(GatedConv1d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm1d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock1d(dim_in=curr_dim, dim_out=curr_dim, kernel_size=3, stride=1, padding=1))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose1d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm1d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(GatedConv1d(curr_dim, scale, kernel_size=7, stride=1, padding=3))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1)
        c = c.repeat(1, 1, x.size(2))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminatorf0(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=256, conv_dim=64, repeat_num=5, num_speakers=6, scale=10):
        super(Discriminatorf0, self).__init__()
        layers = []
        layers.append(GatedConv1d(scale, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(GatedConv1d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(input_size / np.power(2, repeat_num))
        # kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        # kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = GatedConv1d(curr_dim, 1, kernel_size=kernel_size, stride=1, padding=0) # padding should be 0
        self.conv_clf_spks = GatedConv1d(curr_dim, num_speakers, kernel_size=kernel_size, stride=1, padding=0)  # for num_speaker
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(device)
    train_loader = get_loader('./data/mc/train', 8, 'train', num_workers=1)
    data_iter = iter(train_loader)
    G_mc = Generator().to(device)
    D_mc = Discriminator().to(device)
    G_f0 = Generatorf0().to(device)
    D_f0 = Discriminatorf0().to(device)
    for i in range(10):
        mc_real, lf0, spk_label_org, spk_acc_c_org = next(data_iter)
        mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
        mc_real = mc_real.to(device)                         # Input mc.
        spk_label_org = spk_label_org.to(device)             # Original spk labels.
        spk_acc_c_org = spk_acc_c_org.to(device)             # Original spk acc conditioning.
        mc_fake = G_mc(mc_real, spk_acc_c_org)
        print(mc_fake.size())
        out_src, out_cls = D_mc(mc_fake)
        f0_fake = G_f0(lf0, spk_acc_c_org)
        print(f0_fake.size())
        out_src, out_cls = D_f0(f0_fake)