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
        
        # Model type specific init: create the various building blocks (in a dict)
        if self.model_type == 'Linear':
            blocks = self.init_Linear()
        elif self.model_type == 'ParMlp':
            blocks = self.init_ParMlp()
        elif self.model_type == 'FusionCnn':
            blocks = self.init_FusionCnn()
        else:
            raise NameError('Wrong model type')
        
        # Send building blocks to devices
        # Note: self cannot be used in init_Linear() for example, as object not initialized yet. So, use of a dict(): blocks
        for block_name, block in blocks.items():
            dev = self.devices[int(block_name[-1])]
            if str(type(block)) == "<class 'torch.nn.modules.container.ModuleList'>":
                block = nn.ModuleList([b.to(dev) for b in block])
            else:
                block = block.to(dev)
            setattr(self, block_name, block)

    def init_Linear(self):
        """Building blocks of the multilinear regression model"""

        blocks = dict()
        blocks['fc_0'] = linear_block(self.n_wind_channels)
        blocks['fc_1'] = linear_block(self.n_wind_channels)
        return blocks

    def init_ParMlp(self):
        """Building blocks of the simple neural network model (MLP)"""

        blocks = dict()
        blocks['fc_0'] = mlp_block(self.n_wind_channels)
        blocks['fc_1'] = mlp_block(self.n_wind_channels)
        return blocks
    
    def init_FusionCNN(self):
        blocks = dict()
        # Manual parameters defining the architecture
        self.res_lr = 2500  # m
        self.res_hr = 100
        self.size_in_lr = [16, 16]  # pixel
        self.size_ori_hr = [400, 400]
        self.size_in_hr_full = [77, 77]
        self.size_in_hr_zoom = [77, 77]
        self.n_channel_out_lr_full = 32
        self.n_channel_out_lr_zoom = 32
        self.n_channel_out_hr_full = 64
        self.n_channel_out_hr_zoom = 64
        self.n_flat_out_fusion_full = 1
        self.n_flat_out_fusion_zoom = 1
        self.n_channel_out_fusion_full = 512
        self.n_channel_out_fusion_zoom = 512
        self.n_out_fc_full = 128
        self.n_out_fc_zoom = 128

        # Parameters computed from values above
        self.n_channel_ctr_hr_full = int((0.25 + 0.5) * self.n_channel_out_hr_full * self.n_var_2Dhr)
        self.n_channel_ctr_hr_zoom = int((0.25 + 0.5) * self.n_channel_out_hr_zoom * self.n_var_2Dhr)
        self.n_channel_in_fusion_full = self.n_var_2Dlr * self.n_channel_out_lr_full + \
            self.n_var_2Dhr * self.n_channel_out_hr_full
        self.n_channel_in_fusion_zoom = self.n_var_2Dlr * self.n_channel_out_lr_zoom + \
            self.n_var_2Dhr * self.n_channel_out_hr_zoom
        self.n_input_fc_full = self.n_channel_out_fusion_full * self.n_flat_out_fusion_full
        self.n_input_fc_zoom = self.n_channel_out_fusion_zoom * self.n_flat_out_fusion_zoom
        self.n_input_fc_final = self.n_input_1D + self.n_channel_ctr_hr_full + self.n_channel_ctr_hr_zoom + \
            self.n_out_fc_full + self.n_out_fc_zoom + \
            int(self.n_channel_out_fusion_full * (0.5 + 0.25 + 0.125) + self.n_channel_in_fusion_full) + \
            int(self.n_channel_out_fusion_zoom * (0.5 + 0.25 + 0.125) + self.n_channel_in_fusion_zoom)
        

        # Indices to crop/resample the zoom domain out of the full domain for 2Dlr data: CANNOT TO BE AUTOMATIZED EASILY,
        # because 399, 19 and 77 were chosen intentionally (and chosen to be odd for CNNs)
        # The values below perfectly mimic the central crop of 77x77 from the 399x399 data because:
        # 77/399 = 11/57 = (19*11)/(19*57)
        # 1st step: centrally crop the 2Dlr data (19x19) to 11x11
        row_width0 = 11
        self.zoom_lr_row_start0 = int(0.5 * (self.size_in_lr[0] - row_width0))  # = 4
        self.zoom_lr_row_stop0 = self.zoom_lr_row_start0 + row_width0
        col_width0 = 11
        self.zoom_lr_col_start0 = int(0.5 * (self.size_in_lr[1] - col_width0))  # = 4
        self.zoom_lr_col_stop0 = self.zoom_lr_col_start0 + col_width0
        # 2nd step: resample this crop from 11x11 to 57x57
        self.zoom_lr_newsize_row = 57
        self.zoom_lr_newsize_col = 57
        # 3rd step: centrally crop this data to keep 19x19
        self.zoom_lr_row_start1 = int(0.5 * (self.zoom_lr_newsize_row - self.size_in_lr[0]))  # = 19 (coincidence that same value as size of 2Dlr)
        self.zoom_lr_row_stop1 = self.zoom_lr_row_start1 + self.size_in_lr[0]
        self.zoom_lr_col_start1 = int(0.5 * (self.zoom_lr_newsize_col - self.size_in_lr[1]))  # = 19 (same)
        self.zoom_lr_col_stop1 = self.zoom_lr_col_start1 + self.size_in_lr[1]


        # Create each block in the architecture
        for s in ['0', '1']:
            dev = self.devices[int(s)]
            blocks['list_conv_lr_full_' + s] = nn.ModuleList([conv_lr_full_block(n, self.n_channel_out_lr_full, dev)
                                                              for n in self.n_channel_var_2Dlr])
            blocks['list_conv_lr_zoom_' + s] = nn.ModuleList([conv_lr_zoom_block(n, self.n_channel_out_lr_zoom, dev)
                                                              for n in self.n_channel_var_2Dlr])
            blocks['list_conv_hr_full_' + s] = nn.ModuleList([conv_hr_full_block(n, self.n_channel_out_hr_full, dev)
                                                              for n in self.n_channel_var_2Dhr])
            blocks['list_conv_hr_zoom_' + s] = nn.ModuleList([conv_hr_zoom_block(n, self.n_channel_out_hr_zoom, dev)
                                                              for n in self.n_channel_var_2Dhr])
            blocks['conv_fusion_full_' + s] = conv_fusion_full_block(self.n_channel_in_fusion_full,
                                                                     self.n_channel_out_fusion_full, dev)
            blocks['conv_fusion_zoom_' + s] = conv_fusion_zoom_block(self.n_channel_in_fusion_zoom,
                                                                     self.n_channel_out_fusion_zoom, dev)
            blocks['fc_full_' + s] = fc_full_block(self.n_input_fc_full, dev)
            blocks['fc_zoom_' + s] = fc_zoom_block(self.n_input_fc_zoom, dev)
            blocks['fc_final_' + s] = fc_final_block(self.n_input_fc_final, dev)

        return blocks



#Building Blocks
def linear_block(n_input):
    """Used for a multilinear regression model instead of Wind-Topo"""

    return nn.Linear(n_input, 1)


def mlp_block(n_input):
    """Used for a simple neural network model instead of Wind-Topo"""

    return nn.Sequential(
        nn.Linear(n_input, 32),
        nn.SiLU(),
        nn.Linear(32, 16),
        nn.SiLU(),
        nn.Linear(16, 8),
        nn.SiLU(),
        nn.Linear(8, 4),
        nn.SiLU(),
        nn.Linear(4, 1))


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
    range_u = torch.arange(m, device=u.device, dtype=t_type)[None, None, :]
    range_u = range_u.repeat(n_sample, 1, 1)  # Not like np.repeat (in pytorch, it is like np.tile)
    range_v = torch.arange(n, device=v.device, dtype=t_type)[None, :, None]
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


# Tip: out_sz = (in_sz + 2*pad - f_sz )/stride + 1 for conv
#      out_sz = (in_sz + 2*pad - (f_sz-1)*dilation  - 1)/stride + 1 for pooling


def conv_lr_full_block(n_channel_in, n_channel_out, device):
    """Conv block for 3D (u,v,w',dtheta_dz) or 2D (qs, z_nwp) low-res (19x19) full domain data from NWP model"""

    return nn.Sequential(
        nn.Conv2d(n_channel_in, 16, kernel_size=3, stride=1, padding=0),  # 19*19 to 17*17
        nn.SiLU(),
        nn.Conv2d(16, n_channel_out, kernel_size=3, stride=1, padding=0),  # 17*17 to 15*15
        nn.SiLU()).to(device)


def conv_lr_zoom_block(n_channel_in, n_channel_out, device):
    """Conv block for 3D (u,v,w'',dtheta_dz) or 2D (qs, z_nwp) low-res (19x19) zoom domain data from NWP model"""

    return nn.Sequential(
        nn.Conv2d(n_channel_in, 16, kernel_size=3, stride=1, padding=0),  # 19*19 to 17*17
        nn.SiLU(),
        nn.Conv2d(16, n_channel_out, kernel_size=3, stride=1, padding=0),  # 17*17 to 15*15
        nn.SiLU()).to(device)


class conv_hr_full_block(nn.Module):

    """Conv block for 3D (now, none) or 2D (all topo descriptors) high-res (77x77) full domain data from topo desciptors"""

    def __init__(self, n_channel_in, n_channel_out, device):

        super(conv_hr_full_block, self).__init__()

        self.n_channel_in = n_channel_in
        self.n_channel_out = n_channel_out
        self.device = device

        self.conv0 = nn.Conv2d(self.n_channel_in, 16, kernel_size=3, stride=1, padding=0).to(self.device)  # 77*77 to 75*75
        self.f0 = nn.SiLU().to(self.device)
        self.avg0 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0).to(self.device)  # 75*75 to 37*37

        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0).to(self.device)  # 37*37 to 35*35
        self.f1 = nn.SiLU().to(self.device)
        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0).to(self.device)  # 35*35 to 17*17

        self.conv2 = nn.Conv2d(32, self.n_channel_out, kernel_size=3, stride=1, padding=0).to(self.device)  # 17*17to 15*15
        self.f2 = nn.SiLU().to(self.device)

    def forward(self, input, u, v):

        z0 = self.conv0(input)
        a0 = self.f0(z0) * direction_mask(u, v, z0.shape[2], z0.shape[3])

        # Extract central values
        i_row0, i_col0 = int(round(0.5 * (a0.shape[2] - 1))), int(round(0.5 * (a0.shape[3] - 1)))
        y0_out = a0[:, :, i_row0, i_col0]

        y0 = self.avg0(a0)

        z1 = self.conv1(y0)
        a1 = self.f1(z1)

        # Extract central values
        i_row1, i_col1 = int(round(0.5 * (a1.shape[2] - 1))), int(round(0.5 * (a1.shape[3] - 1)))
        y1_out = a1[:, :, i_row1, i_col1]

        y1 = self.avg1(a1)

        z2 = self.conv2(y1)
        y2 = self.f2(z2)

        return (y2, torch.cat((y0_out, y1_out), 1))


class conv_hr_zoom_block(conv_hr_full_block):

    """Conv block for 3D (now, none) or 2D (all topo descriptors) high-res (77x77) zoom domain data from topo desciptors"""

    pass


class conv_fusion_full_block(nn.Module):

    """Conv block on the aggregated outputs of the CNNs that treated full domain data"""

    def __init__(self, n_channel_in, n_channel_out, device):

        super(conv_fusion_full_block, self).__init__()

        self.n_channel_in = n_channel_in
        self.n_channel_out = n_channel_out
        self.device = device

        self.conv0 = nn.Conv2d(self.n_channel_in, 64, kernel_size=3, stride=1, padding=0).to(self.device)  # 15*15 to 13*13
        self.f0 = nn.SiLU().to(self.device)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0).to(self.device)  # 13*13 to 11*11
        self.f1 = nn.SiLU().to(self.device)
        self.avg1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0).to(self.device)  # 11*11 to 5*5

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0).to(self.device)  # 5*5 to 3*3
        self.f2 = nn.SiLU().to(self.device)

        self.conv3 = nn.Conv2d(256, self.n_channel_out, kernel_size=3, stride=1, padding=0).to(self.device)  # 3*3 to 1*1
        self.f3 = nn.SiLU().to(self.device)

    def forward(self, input):

        # Extract central values
        i_row, i_col = int(round(0.5 * (input.shape[2] - 1))), int(round(0.5 * (input.shape[3] - 1)))
        yinput_out = input[:, :, i_row, i_col]

        z0 = self.conv0(input)
        y0 = self.f0(z0)

        # Extract central values
        i_row0, i_col0 = int(round(0.5 * (y0.shape[2] - 1))), int(round(0.5 * (y0.shape[3] - 1)))
        y0_out = y0[:, :, i_row0, i_col0]

        z1 = self.conv1(y0)
        a1 = self.f1(z1)

        # Extract central values
        i_row1, i_col1 = int(round(0.5 * (a1.shape[2] - 1))), int(round(0.5 * (a1.shape[3] - 1)))
        y1_out = a1[:, :, i_row1, i_col1]

        y1 = self.avg1(a1)

        z2 = self.conv2(y1)
        y2 = self.f2(z2)

        # Extract central values
        i_row2, i_col2 = int(round(0.5 * (y2.shape[2] - 1))), int(round(0.5 * (y2.shape[3] - 1)))
        y2_out = y2[:, :, i_row2, i_col2]

        z3 = self.conv3(y2)
        y3 = self.f3(z3)

        y3 = y3[:, :, 0, 0]  # Removes last 2 dims (1x1). Required for input to fully connected layers
        # WARNING: If the architecture here is changed such that y3 has a shape:
        # (n_sample, n_channel, x, y) with x and y !=1, then y3 needs to be flattened, and
        # self.n_flat_out_fusion_full should = x*y (also adapt the zoom branch)

        return (y3, torch.cat((yinput_out, y0_out, y1_out, y2_out), 1))


class conv_fusion_zoom_block(conv_fusion_full_block):

    """Conv block on the aggregated outputs of the CNNs that treated zoom domain data"""

    pass


def fc_full_block(n_input, device):
    """Fully connected NN block on the flattened outputs of the fusion CNN (full)"""

    return nn.Sequential(
        nn.Linear(n_input, 512),
        nn.SiLU(),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.Linear(256, 128),
        nn.SiLU()).to(device)


def fc_zoom_block(n_input, device):
    """Fully connected NN block on the flattened outputs of the fusion CNN (zoom)"""

    return fc_full_block(n_input, device)


def fc_final_block(n_input, device):
    """Fully connected NN block on the aggregated outputs of the above FCNNs, pointwise and central data"""

    return nn.Sequential(
        nn.Linear(n_input, 512),
        nn.SiLU(),
        nn.Linear(512, 128),
        nn.SiLU(),
        nn.Linear(128, 32),
        nn.SiLU(),
        nn.Linear(32, 8),
        nn.SiLU(),
        nn.Linear(8, 1)).to(device)