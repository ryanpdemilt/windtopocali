# Installed packages

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from numpy import float32 as np_f32


torch.pi = torch.acos(torch.zeros(1)).item() * 2  # pi value as a tensor

class WindTopoCali(nn.Module):
    def __init__(self,
                 n_weather_channels,
                 wind_in_size,
                 n_topo_channels,
                 topo_in_size,
                 output_channels,
                 model_type,
                 device=None
                ):
        self.n_weather_channels = n_weather_channels
        self.wind_in_size = wind_in_size
        self.n_topo_channels = n_topo_channels
        self.topo_in_size = topo_in_size
        self.output_channels = output_channels
        self.model_type = model_type
        self.device = device

        if self.device is None:
            self.device = 'cpu'
        
        self.init_FusionCNN()
    
    def init_FusionCNN(self):
        self.n_channel_out_lr_encode = 64
        self.n_channel_out_hr_encode = 64

        self.n_channel_in_lr_fusion = self.n_channel_out_lr_encode * 2
        self.n_channel_in_hr_fusion = self.n_channel_out_hr_encode * 2

        self.n_channel_fusion_out = 128
        self.out_fc_dim = self.n_channel_fusion_out * 2


        self.lr_weather_net = resnet8(in_channels=self.n_weather_channels,output_dim=self.n_channel_out_lr_encode)
        self.hr_weather_net = resnet8(in_channels=self.n_weather_channels,output_dim=self.n_channel_out_hr_encode)

        self.lr_topo_net = resnet8(in_channels=self.n_topo_channels,output_dim=self.n_channel_out_lr_encode)
        self.hr_topo_net = resnet8(in_channels=self.n_topo_channels,output_dim=self.n_channel_out_hr_encode)

        self.hr_fusion_conv = nn.Sequential([
            nn.Conv2d(self.n_channel_in_hr_fusion,64,kernel_size=3,stride=1,padding=0),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(64,128,kernel_size=3,stride=1,paddin=0),
            nn.SiLU(),
            nn.Conv2d(128,self.n_channel_fusion_out,kernel_size=3,stride=1,padding=0)
        ])

        self.lr_fusion_conv = nn.Sequential([
            nn.Conv2d(self.n_channel_in_lr_fusion,64,kernel_size=3,stride=1,padding=0),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(64,128,kernel_size=3,stride=1,paddin=0),
            nn.SiLU(),
            nn.Conv2d(128,self.n_channel_fusion_out,kernel_size=3,stride=1,padding=0)
        ])

        
        self.fc_final_block = nn.Sequential([
            nn.Linear(self.out_fc_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 8),
            nn.SiLU(),
            nn.Linear(8, 2)
        ])

    def forward(self,x):

        rtma = x['rtma']
        rtma_lr = x['rtma_lr']

        topo = x['topo']
        topo_lr = x['topo_lr']

        lr_weather_encode = self.lr_weather_net(rtma_lr)
        hr_weather_encode = self.hr_weather_net(rtma)

        lr_topo_encode = self.lr_topo_net(topo_lr)
        hr_topo_encode = self.hr_topo_net(topo)

        lr_encode = torch.cat([lr_weather_encode,lr_topo_encode],dim=-1)
        lr_encode = self.lr_fusion_conv(lr_encode)

        hr_encode = torch.cat([hr_weather_encode,hr_topo_encode],dim=-1)
        hr_encode = self.hr_fusion_conv(hr_encode)

        fusion_encode = torch.cat([lr_encode,hr_encode],dim=-1)
        out = self.fc_final_block(fusion_encode)

        return out


def direction_mask(u, v, n, m):
    """Returns a mask with 0s on the leeward side (from the center of the domain)

    u and v are (n_sample, 1, 1) pytorch tensors corresponding to the center of the domain
    n and m are nb of rows and columns of the desired mask
    """

    # Put u and v in [0, 1] and inverse u
    n_sample = u.shape[0]
    vel = torch.sqrt(u**2 + v**2)
    ind = vel == 0
    u2 = - u / vel
    v2 = v / vel
    u2[ind], v2[ind] = 0, 0  # With the code below, for vel == 0, mask will be 1

    # Create a matrix of size (n, m) with values = to distance from center in pixels, along the wind direction
    range_u = torch.arange(m, device=u.device)[None, None, :]
    range_u = range_u.repeat(n_sample, 1, 1)  # Not like np.repeat (in pytorch, it is like np.tile)
    range_v = torch.arange(n, device=v.device)[None, :, None]
    range_v = range_v.repeat(n_sample, 1, 1)
    mask = (range_u * u2).repeat(1, n, 1) + (range_v * v2).repeat(1, 1, m)
    mask = mask - mask[:, int(0.5 * (n - 1)), int(0.5 * (m - 1))][:, None, None]  # From the central pixel

    # Binarize this matrix to 0 and val, with val to create a margin from the center of the matrix
    val = - 0.1 * (m + n)  # Treshold value for margin from center: 0.1x(21+21)km=4.2km for full domain and 800m for zoom
    ind_1 = mask >= 0
    ind_0 = mask < val
    mask[ind_0] = val
    mask[ind_1] = 1

    # 0 to 1 with a continuous transition between the 2 values
    mask = (mask - val) / (1 - val)  # {0, val} to {0, 1}
    mask = torch.sin(mask * 0.5 * torch.pi)**2  # Ensures continuity between 0 and 1
    mask = mask[:, None, :, :]  # 4D to match the dims in CNN

    return mask



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        return self.relu(out)
class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels = 3,num_classes=512):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution and max pooling layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the layers dynamically based on the input configuration
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels :
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet8(in_channels=3,output_dim=512):
    return ResNet(ResidualBlock,[1,1,1,1],in_channels=in_channels,num_classes=output_dim)
