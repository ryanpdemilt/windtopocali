import torch
import torch.nn as nn
from torch.nn.modules.module import Module

def weights_init(wt_model):
    """"Randomly initializes model parameters with Xavier method"""
    if isinstance(wt_model, nn.Conv2d):
        nn.init.xavier_uniform_(wt_model.weight.data, gain=1)
    if isinstance(wt_model, nn.Linear):
        nn.init.xavier_uniform_(wt_model.weight.data, gain=1)

# %% Building blocks

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

# %% Architecture


class WindTopo(nn.Module):

    """ This class allows to create, train and use the model Wind-Topo to predict u and v at any given point.

    As the model is an ongoing project, it is easy to change its inputs (e.g., use more/different topo descriptors or NWP data).
    So, information like the number of input data (nb of variables and nb of layers) has to be provided at initialization.
    This class uses the various building blocks defined outside this class in the module windtopo.
    The current version of the architecture of Wind-Topo is named FusionCnn. Different architectures can ba added in the class.
    The initialization (__init__()) is composed of a part that is common to FusionCNN and to the 2 other simpler models
    used for comparison (multilinear regression and neural network) and a part that is specific to each model.

    The forward() method calls a "forward" method that is specific to each model.
    For predictions, the method predict() is used and calls either the standard "no-split" version
    (which is basically a normal forward()) or, for FusionCnn, a "split" version that can be much faster as it reuses the outputs
    of the convolutions performed on data that does not change at every time step (like topo descripors at high-res
    if the directional mask is not used).
    To realize this optional "split" and avoid redundancy in the code, foward_FusionCnn() is simply composed of a method that deal
    with high-res data first: forward_conv_hr_only() and a method doing the rest: forward_rest().
    As forward_conv_hr_only() can be used on specific varaiables / layers, temporally static data can be preprocessed and stored
    using this method. The method forward_split() then uses this data and combines it with the processing of the remaining
    (temporally dynamic) information. In the current version, the split cannot be used because the directional mask changes
    all the model inputs at every time step.

    FusionCnn is composed of 2 branches, preciting u and v respectively, and denoted xxx_0 and xxx_1 in the code.
    If MULTI_GPU is set to True, then all calculations xxx_0 will be done on GPU #0 and all xxx_1 will be done on GPU # 1.
    Otherwise, everything is done on GPU # 0 (or CPU if USE_GPU is set to False).
    For training in MULTI_GPU mode, predicted v is sent to GPU # 0 to allow the custom loss function to deal with u and v.
    """

    def __init__(self, n_channel_var, model_type, model_static_info):
        """Instantiate the class using static info about the model inputs

        "n_channel_va"r is a dict with keys ['1D', '2Dlr', '2Dhr']
            Those 3 variables are lists containing the number of layers of each type of data the model will treat.
        Warning: New topo descriptors are computed on the fly so the number of inputs in x (argument of forward()) is not
        the same as what n_channel_var provides. The latter must remove the input descriptors that are not used by CNNs
        (e.g., slope and aspect) and must include the new topo descriptors (e.g., uE+, uE-, ...)

        E.g., if x['2Dhr_full'] has 3 layers (z, slope, aspect), and 6 new single-layer descriptors are computed
        and if (slope, aspect) are discarded, then: n_channel_var['2Dhr] = [1, 1, 1, 1, 1, 1, 1]
        For the same example if x['1D'] is composed of u(5), v(5), w'(5), dtheta_dz(4), qs(1), time(4), slope(1), aspect(1)
        then n_channel_var['1D] = [5, 5, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1]

        "model_type" is a string to choose which model architecture to use, among: ['Linear', 'ParMlp', 'FusionCnn']
        "model_static_info" is a dict with keys:
            'ind_dir_u_1D' and 'ind_dir_v_1D': int indices to retreave the u and v values in x['1D']
                (needed to compute the wind direction.)
            'new_topo_info': dict containing info needed to compute and integrate on the fly the new topo descriptors
        """

        super(WindTopo, self).__init__()

        # Info needed to create the architecture
        self.n_input_1D = sum(n_channel_var['1D'])
        self.n_var_2Dlr = len(n_channel_var['2Dlr'])
        self.n_channel_var_2Dlr = n_channel_var['2Dlr']
        self.n_var_2Dhr = len(n_channel_var['2Dhr'])
        self.n_channel_var_2Dhr = n_channel_var['2Dhr']
        self.model_type = model_type
        self.ind_dir_u_1D = model_static_info['ind_dir_u_1D']
        self.ind_dir_v_1D = model_static_info['ind_dir_v_1D']
        self.new_topo_info = model_static_info['new_topo_info']

        # Indices in x['2Dhr'] delimiting the inputs to each conv_hr block and index of associated convnets
        # --> needed because of possible split of processing (froward_split())
        tmp = [[], [0], []]
        for i in range(self.n_var_2Dhr):
            tmp[0].append(i)   # Index of convnet
            tmp[1].append(tmp[1][-1] + self.n_channel_var_2Dhr[i])  # Start index in x['2Dr']
            tmp[2].append(tmp[1][-1])  # Stop index
        tmp[1].pop(-1)
        self.ind_2Dhr_treat = [[tmp[0][i], tmp[1][i], tmp[2][i]] for i in range(len(tmp[0]))]

        # Similar information for x['2Dlr] and associated conv_lr blocks
        # Currently no splititng of processing programmed as 2Dlr is for dynamic variables (except z_nwp)
        tmp = [[], [0], []]
        for i in range(self.n_var_2Dlr):
            tmp[0].append(i)   # Index of convnet
            tmp[1].append(tmp[1][-1] + self.n_channel_var_2Dlr[i])  # Start index in x['2Dlr']
            tmp[2].append(tmp[1][-1])  # Stop index
        tmp[1].pop(-1)
        self.ind_2Dlr_treat = [[tmp[0][i], tmp[1][i], tmp[2][i]] for i in range(len(tmp[0]))]

        # Set devices
        self.devices = [0, 0]
        if USE_GPU:
            if MULTI_GPU:
                self.devices[0] = torch.device("cuda:%s" % int(ID_GPU_0))
                self.devices[1] = torch.device("cuda:%s" % int(ID_GPU_1))
            else:
                self.devices = [t_device for _ in range(len(self.devices))]
        else:
            self.devices = [torch.device("cpu") for _ in range(len(self.devices))]

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

    def forward(self, x, model_dyn_info):
        """Returns a model prediction from input dict "x" and some dynamical info.

        Is also used for backpropagation during training.

        "x" is a dict with keys ['1D', '2Dlr', '2Dhr_full', '2Dhr_zoom']
            Those 4 variables are pytorch tensors of size:
            (n_samples, n_layers, n, m) where (n, m) is:
            (1, 1) for x['1D']
            (19, 19) for x['2Dlr']
            (77, 77) for x['2Dhr_full'] and x['2Dhr_zoom']
            n_layers depends on the inputs used for the model,
        "model_dyn_info" is a dict containing info that changes through time. Now only:
            'is_training': bool
            'angle_rot': np.array (n_sample,) of the angle used during random rotation of inputs

        returns "yhat": a dict with currently 1 key only: '1D' containing the prediction (pytorch tensor (n_sample, 2, 1, 1))
        """

        # During training, random rotations of inputs can be used.  Angles can be needed, e.g. to calcule some topo descriptors
        self.is_training = model_dyn_info['is_training']
        if self.is_training:
            self.rot_angles = model_dyn_info['angle_rot']

        yhat = dict()  # Currently each prediction is (u,v) for 1 location. But could evolve, so a dict() for future '2Dhr'
        if self.model_type in ['Linear', 'ParMlp']:
            yhat['1D'] = self.forward_Linear_ParMlp(x)
        elif self.model_type == 'FusionCnn':
            yhat['1D'] = self.forward_FusionCnn(x)
        else:
            raise NameError('Wrong model type')
        if yhat['1D'].ndim == 1:  # When only 1 example in the batch
            yhat['1D'] = yhat['1D'][None, :]
        yhat['1D'] = yhat['1D'][:, :, None, None]
        return yhat

    def update(self, optimizer, criterion, x, y, model_dyn_info):
        """Performs one optimization step during training

        "optimizer": an initialized pytorch optimizer like torch.optim.Adam(...)
        "criterion": an initialized loss function like torch.nn.functional.mse_loss() or windtopo.CustomLoss()
        "x": predictors (inputs) same as argument for forward()
        "y": predictands (ground-truth values), pytorch tensor (n_sample, 2, 1, 1) of u and v values
        "model_dyn_info": same a argument to forward()

        returns "loss": the value of the loss function
        """

        # For loss function, y and yhat must be 2D. So, removes last 2 singleton dims
        y = torch.tensor(y['1D'][:, :, 0, 0], device=self.devices[0], dtype=t_type)
        optimizer.zero_grad()
        yhat = self.forward(x, model_dyn_info)
        yhat = yhat['1D'][:, :, 0, 0]
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

        return loss

    def predict(self, x, model_dyn_info, is_split, ind_2Dhr_treat, outputs_conv_hr_pre):
        """Returns a model prediction yhat cast into a numpy array. Calls the normal or split version of forward().

        "x" and "model_dyn_info" are the same arguments as forward()
        "is_split": bool
        "ind_2Dhr_treat": list of 3 lists
            (0: indices of CNNs to use, 1: starting indices of layers to treat, 2: stopping indices)
        "outputs_conv_hr_pre": tupple containing the outputs of the CNNs on precomputed
            (temporally static) 2Dhr data. (in split mode only)
        """
        with torch.no_grad():
            if is_split & (self.model_type == 'FusionCnn'):  # Split version available for FusionCnn only
                yhat = self.forward_split(x, model_dyn_info, ind_2Dhr_treat, outputs_conv_hr_pre)
            else:
                yhat = self.forward(x, model_dyn_info)
            yhat['1D'] = yhat['1D'].to(torch.device("cpu")).detach().numpy()

        return yhat

    # %% Simple pointwise models

    def init_Linear(self):
        """Building blocks of the multilinear regression model"""

        blocks = dict()
        blocks['fc_0'] = linear_block(self.n_input_1D)
        blocks['fc_1'] = linear_block(self.n_input_1D)
        return blocks

    def init_ParMlp(self):
        """Building blocks of the simple neural network model (MLP)"""

        blocks = dict()
        blocks['fc_0'] = mlp_block(self.n_input_1D)
        blocks['fc_1'] = mlp_block(self.n_input_1D)
        return blocks

    def forward_Linear_ParMlp(self, x):
        """forward specific to multilinear regression model and neural network (MLP)
        "x": dict with only '1D' key containing a pytorch tensor (n_samples, n_layers, 1, 1)

        returns "yhat": model predictions u and v: same as x but size (n_samples, 2, 1, 1)
        """

        x_1D_0 = torch.tensor(x['1D'], device=self.devices[0], dtype=t_type)
        u_0, v_0 = x_1D_0[:, [self.ind_dir_u_1D], :, :], x_1D_0[:, [self.ind_dir_v_1D], :, :]

        new_topo = self.compute_topo(x_1D_0, u_0, v_0)
        x_1D_0 = self.torch_delete_4D(x_1D_0, self.new_topo_info['ind_slope_1D'] + self.new_topo_info['ind_aspect_1D'])
        x_1D_0 = torch.cat((x_1D_0, new_topo), 1)

        x_1D_1 = x_1D_0
        if self.devices[1] != self.devices[0]:
            x_1D_1 = x_1D_1.to(self.devices[1], non_blocking=True, copy=False)

        u = self.fc_0(x_1D_0.squeeze())
        v = self.fc_1(x_1D_1.squeeze())
        if self.devices[1] != self.devices[0]:
            v = v.to(self.devices[0], non_blocking=True, copy=False)

        yhat = torch.cat((u, v), 1)
        return yhat

    # %% Wind-Topo

    def init_FusionCnn(self):
        """Initialization specific to the architecture FusionCnn. Parameters needed for the architecture are defined here.

        Returns "blocks": a dict containing all the building blocks (CNN, FCNN) for branch 0 and 1
        """

        blocks = dict()
        # Manual parameters defining the architecture
        self.res_lr = 1113  # m
        self.res_hr = 53
        self.size_in_lr = [19, 19]  # pixel
        self.size_ori_hr = [399, 399]
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

    def forward_FusionCnn(self, x):
        """ forward specific to FusionCnn architecture.

        "x" is the same as the argument of forward()

        Returns "yhat": model predictions, same as foward()
        """

        self.check_inputs(x)

        # Extracting u and v values used for wind direction
        u, v = x['1D'][:, [self.ind_dir_u_1D], :, :], x['1D'][:, [self.ind_dir_v_1D], :, :]

        # Convolutions on HR data
        outputs_conv_hr = self.forward_conv_hr_only(x['2Dhr_full'], x['2Dhr_zoom'], self.ind_2Dhr_treat, u, v)

        # Prediction
        yhat = self.forward_rest(outputs_conv_hr, x['2Dlr'], x['1D'])

        return yhat

    def check_inputs(self, x):
        """Performs a sanity check on the model inputs "x": type, nb of dims, nb of layers, nb of pixels"""

        sz = x['1D'].shape
        n = self.new_topo_info['delta_n_layer']
        assert((len(sz) == 4) & (x['1D'].dtype == np_f32))
        assert(sz[1:] == (self.n_input_1D - n, 1, 1))
        sz = x['2Dlr'].shape
        assert((len(sz) == 4) & (x['2Dlr'].dtype == np_f32))
        assert(sz[1:] == (sum(self.n_channel_var_2Dlr), self.size_in_lr[0], self.size_in_lr[1]))
        sz = x['2Dhr_full'].shape
        assert((len(sz) == 4) & (x['2Dhr_full'].dtype == np_f32))
        assert(sz[1:] == (sum(self.n_channel_var_2Dhr) - n, self.size_in_hr_full[0], self.size_in_hr_full[1]))
        sz = x['2Dhr_zoom'].shape
        assert((len(sz) == 4) & (x['2Dhr_zoom'].dtype == np_f32))
        assert(sz[1:] == (sum(self.n_channel_var_2Dhr) - n, self.size_in_hr_zoom[0], self.size_in_hr_zoom[1]))

    def forward_conv_hr_only(self, x_2Dhr_full_0, x_2Dhr_zoom_0, ind_2Dhr_treat, u, v):
        """First step for the foward of FusionCnn: treat 2Dhr inputs for selected layers only

        "x_2Dhr_full_0" and "x_2Dhr_zoom_0": np array contained in x['2D_hr_full'] and x['2D_hr_zoom']
        "ind_2Dhr_treat": list of 3 lists
            (0: indices of CNNs to use, 1: starting indices of layers to treat, 2: stopping indices)
        "u" and "v": (n_sample, 1, 1, 1) np array of u and v values used to compute wind direction

        Returns "outputs_conv_hr": tupple of the outputs of the CNNs
        """

        # Convert branch 0 data from np array to pytorch tensors and send data to correct device
        x_2Dhr_full_0 = torch.tensor(x_2Dhr_full_0, device=self.devices[0], dtype=t_type)
        x_2Dhr_zoom_0 = torch.tensor(x_2Dhr_zoom_0, device=self.devices[0], dtype=t_type)
        u_0 = torch.tensor(u, device=self.devices[0], dtype=t_type)
        v_0 = torch.tensor(v, device=self.devices[0], dtype=t_type)

        # Operations on topo descriptors: the 2 layers of slope and aspect will be replaced by 6 layers
        #    On full data
        new_topo_full = self.compute_topo(x_2Dhr_full_0, u_0, v_0)
        x_2Dhr_full_0 = self.torch_delete_4D(x_2Dhr_full_0,
                                             self.new_topo_info['ind_slope_2Dhr'] + self.new_topo_info['ind_aspect_2Dhr'])
        x_2Dhr_full_0 = torch.cat((x_2Dhr_full_0, new_topo_full), 1)
        #    On zoom data
        new_topo_zoom = self.compute_topo(x_2Dhr_zoom_0, u_0, v_0)
        x_2Dhr_zoom_0 = self.torch_delete_4D(x_2Dhr_zoom_0,
                                             self.new_topo_info['ind_slope_2Dhr'] + self.new_topo_info['ind_aspect_2Dhr'])
        x_2Dhr_zoom_0 = torch.cat((x_2Dhr_zoom_0, new_topo_zoom), 1)

        # Copy the data from branch 0 to branch 1 and send to correct device
        x_2Dhr_full_1 = x_2Dhr_full_0
        x_2Dhr_zoom_1 = x_2Dhr_zoom_0
        u_1, v_1 = u_0, v_0
        if self.devices[1] != self.devices[0]:
            x_2Dhr_full_1 = x_2Dhr_full_1.to(self.devices[1], non_blocking=True, copy=False)
            x_2Dhr_zoom_1 = x_2Dhr_zoom_1.to(self.devices[1], non_blocking=True, copy=False)
            u_1 = u_1.to(self.devices[1], non_blocking=True, copy=False)
            v_1 = v_1.to(self.devices[1], non_blocking=True, copy=False)
        u_0, v_0 = torch.squeeze(u_0, 3), torch.squeeze(v_0, 3)  # u, v were 4D for compute_topo. Now 3D for direct. masks
        u_1, v_1 = torch.squeeze(u_1, 3), torch.squeeze(v_1, 3)

        # CNNs treating the selected topo descriptors
        #    For full data
        hr_0_full = [self.list_conv_hr_full_0[ind[0]](x_2Dhr_full_0[:, ind[1]:ind[2], :, :], u_0, v_0)
                     for ind in ind_2Dhr_treat]
        hr_1_full = [self.list_conv_hr_full_1[ind[0]](x_2Dhr_full_1[:, ind[1]:ind[2], :, :], u_1, v_1)
                     for ind in ind_2Dhr_treat]
        #    Reorginize the outputs of the CNN blocks (tupples of CNN outputs and central values)
        conv_out_full_0 = torch.cat([t[0] for t in hr_0_full], 1)
        conv_out_full_1 = torch.cat([t[0] for t in hr_1_full], 1)
        conv_ctr_full_0 = torch.cat([t[1] for t in hr_0_full], 1)
        conv_ctr_full_1 = torch.cat([t[1] for t in hr_1_full], 1)
        #    For zoom data
        hr_0_zoom = [self.list_conv_hr_zoom_0[ind[0]](x_2Dhr_zoom_0[:, ind[1]:ind[2], :, :], u_0, v_0)
                     for ind in ind_2Dhr_treat]
        hr_1_zoom = [self.list_conv_hr_zoom_1[ind[0]](x_2Dhr_zoom_1[:, ind[1]:ind[2], :, :], u_1, v_1)
                     for ind in ind_2Dhr_treat]
        #    Reorginize the outputs of the CNN blocks (tupples of CNN outputs and central values)
        conv_out_zoom_0 = torch.cat([t[0] for t in hr_0_zoom], 1)
        conv_out_zoom_1 = torch.cat([t[0] for t in hr_1_zoom], 1)
        conv_ctr_zoom_0 = torch.cat([t[1] for t in hr_0_zoom], 1)
        conv_ctr_zoom_1 = torch.cat([t[1] for t in hr_1_zoom], 1)

        outputs_conv_hr = (conv_out_full_0, conv_out_full_1, conv_out_zoom_0, conv_out_zoom_1,
                           conv_ctr_full_0, conv_ctr_full_1, conv_ctr_zoom_0, conv_ctr_zoom_1)
        return outputs_conv_hr

    def forward_rest(self, outputs_conv_hr, x_2Dlr, x_1D):
        """Second step for the foward of FusionCnn: calculation which has to be done for every data point (time series)

        "outputs_conv_hr": tupple of the outputs of the CNNs
        "x_2Dlr" and "x_1D": np arrays contained in x['2D_lr'] and x['1D']

        Returns "yhat": predictions of u and v (pytorch tensor (n_sample, 2))
        """

        # Convert branch 0 data from np array to pytorch tensor and send data to correct device
        x_1D_0 = torch.tensor(x_1D, device=self.devices[0], dtype=t_type)
        x_2Dlr_full_0 = torch.tensor(x_2Dlr, device=self.devices[0], dtype=t_type)
        u_0, v_0 = x_1D_0[:, [self.ind_dir_u_1D], :, :], x_1D_0[:, [self.ind_dir_v_1D], :, :]

        # Operations on topo descriptors: the 2 layers of slope and aspect will be replaced by 6 layers
        new_topo = self.compute_topo(x_1D_0, u_0, v_0)
        x_1D_0 = self.torch_delete_4D(x_1D_0, self.new_topo_info['ind_slope_1D'] + self.new_topo_info['ind_aspect_1D'])
        x_1D_0 = torch.cat((x_1D_0, new_topo), 1)
        x_1D_0 = x_1D_0[:, :, 0, 0]  # Removes last 2 singleton dims for input of final fully connected NN

        # Copy the data from branch 0 to branch 1 and send to correct device
        x_1D_1 = x_1D_0
        x_2Dlr_full_1 = x_2Dlr_full_0
        if self.devices[1] != self.devices[0]:
            x_1D_1 = x_1D_1.to(self.devices[1], non_blocking=True, copy=False)
            x_2Dlr_full_1 = x_2Dlr_full_1.to(self.devices[1], non_blocking=True, copy=False)

        # Generate the zoom data from the full data
        #    Crop centrally (1st step), then resample to higher resolution (2nd step)
        x_2Dlr_zoom_0 = torch.nn.functional.interpolate(x_2Dlr_full_0[:, :, self.zoom_lr_row_start0:self.zoom_lr_row_stop0,
                                                                      self.zoom_lr_col_start0:self.zoom_lr_col_stop0],
                                                        (self.zoom_lr_newsize_row, self.zoom_lr_newsize_col),
                                                        mode='bilinear', align_corners=False)
        #    Crop centrally (3rd step)
        x_2Dlr_zoom_0 = x_2Dlr_zoom_0[:, :, self.zoom_lr_row_start1:self.zoom_lr_row_stop1,
                                      self.zoom_lr_col_start1:self.zoom_lr_col_stop1]

        # Copy the data from branch 0 to branch 1 and send to correct device
        x_2Dlr_zoom_1 = x_2Dlr_zoom_0
        if self.devices[1] != self.devices[0]:
            x_2Dlr_zoom_1 = x_2Dlr_zoom_1.to(self.devices[1], non_blocking=True, copy=False)

        # CNNs for the 2 branches and for full and zoom data on 2Dlr data
        lr_0_full = torch.cat([self.list_conv_lr_full_0[ind[0]](x_2Dlr_full_0[:, ind[1]:ind[2], :, :])
                               for ind in self.ind_2Dlr_treat], 1)
        lr_1_full = torch.cat([self.list_conv_lr_full_1[ind[0]](x_2Dlr_full_1[:, ind[1]:ind[2], :, :])
                               for ind in self.ind_2Dlr_treat], 1)
        lr_0_zoom = torch.cat([self.list_conv_lr_zoom_0[ind[0]](x_2Dlr_zoom_0[:, ind[1]:ind[2], :, :])
                               for ind in self.ind_2Dlr_treat], 1)
        lr_1_zoom = torch.cat([self.list_conv_lr_zoom_1[ind[0]](x_2Dlr_zoom_1[:, ind[1]:ind[2], :, :])
                               for ind in self.ind_2Dlr_treat], 1)

        # Unpack the outputs of the CNNs on the 2Dhr data
        (conv_out_full_0, conv_out_full_1, conv_out_zoom_0, conv_out_zoom_1,
         conv_ctr_full_0, conv_ctr_full_1, conv_ctr_zoom_0, conv_ctr_zoom_1) = outputs_conv_hr
        del outputs_conv_hr

        # Concatenate CNNs outputs from 2Dlr and 2Dhr
        conv_out_full_0 = torch.cat((conv_out_full_0, lr_0_full), 1)
        conv_out_full_1 = torch.cat((conv_out_full_1, lr_1_full), 1)
        conv_out_zoom_0 = torch.cat((conv_out_zoom_0, lr_0_zoom), 1)
        conv_out_zoom_1 = torch.cat((conv_out_zoom_1, lr_1_zoom), 1)

        # Fusion CNNs on these concatenated data
        conv_out_full_0_fused = self.conv_fusion_full_0(conv_out_full_0)
        conv_out_full_1_fused = self.conv_fusion_full_1(conv_out_full_1)
        conv_out_zoom_0_fused = self.conv_fusion_zoom_0(conv_out_zoom_0)
        conv_out_zoom_1_fused = self.conv_fusion_zoom_1(conv_out_zoom_1)

        # Fully connected nets
        u_fusion = self.fc_full_0(conv_out_full_0_fused[0])
        v_fusion = self.fc_full_1(conv_out_full_1_fused[0])
        u_fusion_zoom = self.fc_zoom_0(conv_out_zoom_0_fused[0])
        v_fusion_zoom = self.fc_zoom_1(conv_out_zoom_1_fused[0])
        u_all = self.fc_final_0(torch.cat((u_fusion, u_fusion_zoom, x_1D_0, conv_ctr_full_0,
                                conv_ctr_zoom_0, conv_out_full_0_fused[1], conv_out_zoom_0_fused[1]), 1))
        v_all = self.fc_final_1(torch.cat((v_fusion, v_fusion_zoom, x_1D_1, conv_ctr_full_1,
                                conv_ctr_zoom_1, conv_out_full_1_fused[1], conv_out_zoom_1_fused[1]), 1))

        # Put results on the same device
        if self.devices[1] != self.devices[0]:
            v_all = v_all.to(self.devices[0], non_blocking=True, copy=False)

        # Output
        yhat = torch.cat((u_all, v_all), 1)
        return yhat

    def forward_split(self, x, model_dyn_info, ind_2Dhr_treat, outputs_conv_hr_pre):
        """Version of foward() that can be used for FusionCnn archtitecture, with precomputed outputs from CNNs on 2Dhr data.

        "x" and "model_dyn_info" are the same as the arguments of forward()
        "ind_2Dhr_treat": list of 3 lists
            (0: indices of CNNs to use, 1: starting indices of layers to treat, 2: stopping indices)
            It indicates which 2Dhr data remains to be treated (because the are not temporally static)
        "outputs_conv_hr_pre": tupple of the precomputed outputs of CNNs that need to be merged with the ones computed here

        Returns: "yhat", model predictions, same as foward()
        """

        self.check_inputs(x)

        # Extracting u and v values used for wind direction
        u, v = x['1D'][:, [self.ind_dir_u_1D], :, :], x['1D'][:, [self.ind_dir_v_1D], :, :]

        # Compute the outputs of the convolutions of 2Dhr variables that are time-dependent
        if len(ind_2Dhr_treat[0]) > 0:
            outputs_conv_hr_temp = self.forward_conv_hr_only(x['2Dhr_full'], x['2Dhr_zoom'], ind_2Dhr_treat, u, v)

        # Incorporate the data that was already computed and call forward_rest()
        yhat = dict()
        outputs_conv_hr = []
        if len(ind_2Dhr_treat[0]) > 0:
            for i in range(8):
                if i in [0, 1]:
                    n_channel = self.n_channel_out_hr_full
                elif i in [2, 3]:
                    n_channel = self.n_channel_out_hr_zoom
                elif i in [4, 5]:
                    n_channel = self.n_channel_ctr_hr_full
                elif i in [6, 7]:
                    n_channel = self.n_channel_ctr_hr_zoom
                for i in range(len(ind_2Dhr_treat[0])):
                    n_lay_before = ind_2Dhr_treat[1][0]  # index of the 1st layer -> = nb of layers before insertion
                    outputs_conv_hr[i] = torch.cat((outputs_conv_hr_pre[i][:, :n_lay_before * n_channel, :, :],
                                                    outputs_conv_hr_temp[i],
                                                    outputs_conv_hr_pre[i][:, n_lay_before * n_channel:, :, :]), 1)
            # Convert list to tuple to match argument of forward_rest()
            outputs_conv_hr = tuple(outputs_conv_hr)
            yhat['1D'] = self.forward_rest(outputs_conv_hr, x['2Dlr'], x['1D'])
        else:
            yhat['1D'] = self.forward_rest(outputs_conv_hr_pre, x['2Dlr'], x['1D'])

        # Output
        if yhat['1D'].ndim == 1:  # When only 1 example in the batch
            yhat['1D'] = yhat['1D'][None, :]
        yhat['1D'] = yhat['1D'][:, :, None, None]

        return yhat

    def torch_delete_4D(self, tensor, indices):
        """Removes the layers (dimension 1) from 4D pytorch tensor "tensor" indicated by "indices". """

        mask = torch.ones(tensor.shape[1], dtype=torch.bool)
        mask[indices] = False
        return tensor[:, mask, :, :]

    def compute_topo(self, x, u, v):
        """Generates new topographic descriptors from model inputs

        "x" is a model input, 4D pytorch tensor of shape (n_samples, n_layers, n, m)
            where (n, m) is (1, 1) or (77, 77). "x" is either x['1D'] or x['2Dhr_full'] or x['2Dhr_zoom']
        "u" and "v" are 4D pytorch tensor (n_samples, 1, 1, 1) used to compute wind direction

        Returns: "new_topo", pytorch tensor of shape (n_samples, n_new_topo, n, m)
        """

        if x.shape[-1] == 1:
            ind_slope, ind_aspect = self.new_topo_info['ind_slope_1D'], self.new_topo_info['ind_aspect_1D']
        else:
            ind_slope, ind_aspect = self.new_topo_info['ind_slope_2Dhr'], self.new_topo_info['ind_aspect_2Dhr']
        # Angle between aspect (trigo) and opposite of wind dir (trigo)
        delta = torch.fmod(torch.atan2(-v, -u) + 2 * torch.pi, 2 * torch.pi) - x[:, ind_aspect, :, :]
        #                                             atan in [0 2pi] (carefull, fmod can return neg)
        delta = torch.fmod(delta + 3 * torch.pi, 2 * torch.pi) - torch.pi   # in [-pi pi]

        # Exposure and sheltering coefficients based on the wind/slope angle: alpha
        alpha = torch.atan(torch.tan(x[:, ind_slope, :, :]) * torch.cos(delta))
        sin_alpa = torch.sin(alpha)
        torch_zero = torch.tensor([0], dtype=t_type, device=sin_alpa.device)
        eps_plus = torch.maximum(sin_alpa, torch_zero)
        eps_minus = torch.min(sin_alpa, torch_zero)

        # Deflection angle (such that wind is tangential when slope is 90 deg)
        beta = (0.5 * torch.pi - torch.abs(delta)) * torch.sign(delta) * torch.sin(x[:, ind_slope, :, :])**2

        # Deltas of u and v to make them tangential
        cos_ang = torch.cos(beta) - 1  # -1 because we compute deltas (below)
        sin_ang = torch.sin(beta)
        delta_u_tan = cos_ang * u - sin_ang * v
        delta_v_tan = sin_ang * u + cos_ang * v

        # New topo descriptors
        new_topo = torch.cat((u * eps_plus, u * eps_minus, v * eps_plus, v * eps_minus,
                              delta_u_tan, delta_v_tan), 1)

        return new_topo
