"""
@Time: 2024/6/11
@Auth: Qi Wang
@File: burgers_p2c2.py
@Mails: qi_wang@ruc.edu.cn
@Motto: I work model work.
"""
from mindspore import Tensor, context, Parameter
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore
import mindspore.ops as ops

from mindflow.cell.neural_operators.fno2d import FNOBlock

from mindspore.common.initializer import Zero
# from mindflow.cell.neural_operators.fno2d import SpectralConv2dDft
from mindflow.cell.neural_operators import FNO2D

from mindflow.cell.neural_operators.dft import dft2, idft2
# import dft.dft2, dft.idft2


# from torch.utils.data import Dataset, DataLoader
from mindspore.dataset import GeneratorDataset as Dataset

import numpy as np
import matplotlib.pyplot as plt

import random

# import torch.nn.functional as F

import scipy.io as sio
import time

import gc
import os

#context.set_context(max_call_depth=20000)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#context.set_context(mode=context.PYNATIVE_MODE,
context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target='GPU',
                    device_id=3)

import sys

expand_dims = ops.ExpandDims()
concat_op = ops.Concat()
transpose = ops.Transpose()
mse_loss = nn.MSELoss()
zeros = ops.Zeros()
prints = ops.Print()
concat = mindspore.ops.Concat()
uniformreal = ops.UniformReal(seed=2)
# torch.set_default_dtype(mindspore.float32)
# torch.manual_seed(42)
mindspore.set_seed(42)
np.random.seed(42)
DerFilter = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1 / 12, -8 / 12, 0, 8 / 12, -1 / 12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]
dx_lb_op = [[[[-25 / 12, 48 / 12, -36 / 12, 16 / 12, -3 / 12]]]]
dx_lsecb_op = [[[[-3 / 12, -10 / 12, 18 / 12, -6 / 12, 1 / 12]]]]
dx_rb_op = [[[[3 / 12, -16 / 12, 36 / 12, -48 / 12, 25 / 12]]]]
dx_rsecb_op = [[[[-1 / 12, 6 / 12, -18 / 12, 10 / 12, 3 / 12]]]]
test_op = [[[[3.55, 6.90, 0.9983942, 0.324324, 0.324324],
             [0.32432, 0.2343, 0.34234, 0.5436, 0.7657],
             [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
             [0.32432, 0.2343, 0.34234, 0.5436, 0.7657],
             [0.99, 0.879798, 0.789324732, 0.898, 0.324]]]]
dxx_op = [[[[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]]]
dxx_rb_op = [[[[11 / 12, -56 / 12, 114 / 12, -104 / 12, 35 / 12]]]]
dxx_rseb_op = [[[[-1 / 12, 4 / 12, 6 / 12, -20 / 12, 11 / 12]]]]
dxx_lb_op = [[[[35 / 12, -104 / 12, 114 / 12, -56 / 12, 11 / 12]]]]
dxx_lsecb_op = [[[[11 / 12, -20 / 12, 6 / 12, 4 / 12, -1 / 12]]]]


class FNO2dWithoutGrid(nn.Cell):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 resolution=26,
                 modes=10,
                 channels=12,
                 depths=4,
                 mlp_ratio=2.5,
                 compute_dtype=mstype.float32):
        super().__init__()

        self.modes1 = modes
        self.channels = channels
        # self.fc_channel = int(mlp_ratio * channels)
        self.fc_channel = 2 * resolution
        self.fc0 = nn.Conv2d(in_channels, self.channels, 1, has_bias=True)
        self.layers = depths

        self.fno_seq = nn.SequentialCell()
        for _ in range(self.layers - 1):
            self.fno_seq.append(FNOBlock(self.channels, self.channels, modes1=self.modes1, resolution=resolution,
                                         compute_dtype=compute_dtype))
        self.fno_seq.append(
            FNOBlock(self.channels, self.channels, self.modes1, resolution=resolution, gelu=False,
                     compute_dtype=compute_dtype))

        self.fc1 = nn.Conv2d(self.channels, self.fc_channel, 1, has_bias=True)
        self.fc2 = nn.Conv2d(self.fc_channel, out_channels, 1, has_bias=True)

        # self.grid = Tensor(get_grid_2d(resolution), dtype=mstype.float32)
        # self.concat = ops.Concat(axis=-1)
        self.act = ops.GeLU()

    def construct(self, x: Tensor):
        x = self.fc0(x)

        x = self.fno_seq(x)

        x = self.fc1(x)
        x = self.act(x)
        output = self.fc2(x)
        return output


class SpectralConv2dDft(nn.Cell):
    def __init__(self, in_channels, out_channels, modes1, resolution, compute_dtype=mstype.float32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.resolution = resolution
        self.compute_dtype = compute_dtype

        self.scale = (1. / (in_channels * out_channels))
        w_re = Tensor(self.scale * np.random.rand(in_channels, out_channels, self.modes1, self.modes1),
                      dtype=compute_dtype)
        w_im = Tensor(self.scale * np.random.rand(in_channels, out_channels, self.modes1, self.modes1),
                      dtype=compute_dtype)
        self.w_re = Parameter(w_re, requires_grad=True)
        self.w_im = Parameter(w_im, requires_grad=True)
        self.dft2_cell = dft2(shape=(self.resolution, self.resolution), modes=modes1,
                              compute_dtype=compute_dtype)
        self.idft2_cell = idft2(shape=(self.resolution, self.resolution), modes=modes1, compute_dtype=compute_dtype)

    @staticmethod
    def mul2d(inputs, weights):  # [40,12,8,8] [12,12,8,8]
        weight = weights.expand_dims(0)  # 1,12,12,8
        data = inputs.expand_dims(2)  # 40,12,1,8,8
        out = weight * data  # 40,12,12,8,8
        return out.sum(1)

    def construct(self, x: Tensor):
        x_re = x
        x_im = ops.zeros_like(x_re)
        x_ft_re, x_ft_im = self.dft2_cell((x_re, x_im))

        out_ft_re = \
            self.mul2d(x_ft_re[:, :, :self.modes1], self.w_re) \
            - self.mul2d(x_ft_im[:, :, :self.modes1], self.w_im)
        out_ft_im = \
            self.mul2d(x_ft_re[:, :, :self.modes1], self.w_re) \
            + self.mul2d(x_ft_im[:, :, :self.modes1], self.w_im)

        x, _ = self.idft2_cell((out_ft_re, out_ft_im))
        return x


class MyDataset_bak(Dataset):

    def __init__(self, data_features):
        self.len = len(data_features)
        self.features = data_features

    def __getitem__(self, index):
        feature = self.features[index]
        x = feature[0, ...]
        seq = feature[1:, ...]
        return x, seq

    def __len__(self):
        return self.len
class MyDatasets:

    def __init__(self, data_features):
        self.len = len(data_features)
        self.features = data_features

    def __getitem__(self, index):
        feature = self.features[index]
        x = feature[0:1, ...]
        seq = feature[1:, ...]
        return x, seq

    def __len__(self):
        return self.len

class MyDataset:
    #
    def __init__(self, data_features):
        self.len = len(data_features)
        self.features = data_features

    def __getitem__(self, index):
        feature = self.features[index]
        x = feature[0:1, ...]
        seq = feature[1:, ...]
        return x, seq

    def __len__(self):
        return self.len


class TimeseriesDataset(Dataset):
    def __init__(self, X, seq_len=1):
        self.X = X

        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len)

    def __getitem__(self, index):
        x = self.X[index]
        seq = self.X[index + 1:index + self.seq_len + 1]
        return x, seq


class SpectralConv2d_fast(nn.Cell):
    def __init__(self, in_channels, out_channels, modes1, modes2, compute_dtype=mstype.float32):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.compute_dtype = compute_dtype
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.w_re = Tensor(self.scale * np.random.rand(in_channels, out_channels, self.modes1, self.modes1),
                           dtype=compute_dtype)
        self.w_im = Tensor(self.scale * np.random.rand(in_channels, out_channels, self.modes1, self.modes1),
                           dtype=compute_dtype)
        # self.weights1 = mindspore.Parameter(
        #     self.scale * mindspore.ops.UniformReal(in_channels, out_channels, self.modes1, self.modes2, dtype=self.compute_dtype))
        # self.weights2 = mindspore.Parameter(
        #     self.scale * mindspore.ops.UniformReal(in_channels, out_channels, self.modes1, self.modes2, dtype=self.compute_dtype))

        self.dft2_cell = dft2(shape=(self.resolution, self.resolution), modes=modes1,
                              compute_dtype=self.compute_dtype)
        self.idft2_cell = idft2(shape=(self.resolution, self.resolution), modes=modes1,
                                compute_dtype=self.compute_dtype)

    # Complex multiplication
    # def compl_mul2d(self, input, weights):  # 20,1,6,6,   weights 1,12,6,6 sample  1 1 12 12  weights 1 13 12 12
    #     # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    #     return torch.einsum("bixy,ioxy->boxy", input, weights)
    @staticmethod
    def mul2d(inputs, weights):
        weight = weights.expand_dims(0)
        data = inputs.expand_dims(2)
        out = weight * data
        return out.sum(1)

    def construct(self, x):  # 20,1,25,25
        batchsize = x.shape[0]
        # print(batchsize)
        x_re = x
        x_im = ops.zeros_like(x_re)
        x_ft_re, x_ft_im = self.dft2_cell((x_re, x_im))

        # Multiply relevant Fourier modes
        # out_ft = mindspore.numpy.zeros(batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1)
        out_ft_re = self.mul2d(x_ft_re[:, :, :self.modes1], self.w_re) - self.mul2d(x_ft_im[:, :, :self.modes1],
                                                                                    self.w_im)
        out_ft_im = self.mul2d(x_ft_re[:, :, :self.modes2], self.w_re) + self.mul2d(x_ft_im[:, :, :self.modes2],
                                                                                    self.w_im)
        # Return to physical space
        # x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        x, _ = self.idft2_cell((out_ft_re, out_ft_im))
        # prints(x.shape)
        return x  # 20,12,25,25


class FNO2d1(nn.Cell):
    def __init__(self, modes1, modes2, width):
        super(FNO2d1, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Conv2d(1, self.width, 1)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2dDft(self.width, self.width, self.modes1, 25)
        self.conv1 = SpectralConv2dDft(self.width, self.width, self.modes1, 25)
        self.conv2 = SpectralConv2dDft(self.width, self.width, self.modes1, 25)
        self.conv3 = SpectralConv2dDft(self.width, self.width, self.modes1, 25)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm2d(self.width)
        # self.bn1 = torch.nn.BatchNorm2d(self.width)
        # self.bn2 = torch.nn.BatchNorm2d(self.width)
        # self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Conv2d(self.width, 50, 1)
        self.fc2 = nn.Conv2d(50, 1, 1)
        # self.relu = nn.ReLU()
        self.act = ops.GeLU()

    def construct(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = mindspore.ops.Concat((x, grid), dim=-1)  #x 1 25 25 1
        x = self.fc0(x)  # 1 25 25 20
        # x = x.permute(0, 3, 1, 2) #1, 20 25 25
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        # print(x.shape)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        # x = self.relu(x)
        x = self.act(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)
        #
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.act(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        # x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        # x = self.relu(x)
        x = self.fc2(x)
        return x


class RCNN(nn.Cell):
    def __init__(self, timesteps):
        super(RCNN, self).__init__()
        # self._all_layers = []
        self.u_cor = FNO2D(in_channels=1,
                           out_channels=1,
                           resolution=26,
                           modes=10,
                           channels=12,
                           depths=4,
                           mlp_ratio=4)
        self.v_cor = FNO2D(in_channels=1,
                           out_channels=1,
                           resolution=26,
                           modes=10,
                           channels=12,
                           depths=4,
                           mlp_ratio=4)
        # self.u_cor =FNO2dWithoutGrid()
        # self.v_cor =FNO2dWithoutGrid()
        self.steps = timesteps
        self.filter = Filter()

        # self.cor = FNO2d(modes, modes, width).cuda()
        # self.cor = FNO2d(modes, modes, width)
        # self.vfilter = Filter()
        self.vis = vis
        self.dt = delta_t
        # name = 'recurrent_cell'
        # cell = RecurrentCell()

        # setattr(self, name, self.cell)
        # self._all_layers.append(self.cell)
    def rk(self,h):
        u_prev = h[:, 0:1, ...]
        v_prev = h[:, 1:2, ...]
        # 1,1,25,25 - > 1,25,25,1   0,3,1,2

        u_prev_trans = ops.transpose(u_prev, (0, 2, 3, 1))
        v_prev_trans = ops.transpose(v_prev, (0, 2, 3, 1))
        # ops.print_(u_prev.shape)
        u_cor = self.u_cor(u_prev_trans)
        v_cor = self.v_cor(v_prev_trans)
        u_cor = ops.transpose(u_cor, (0, 3, 1, 2))
        v_cor = ops.transpose(v_cor, (0, 3, 1, 2))
        # ops.print_(u_cor.shape)
        filter_u = self.filter(u_cor)
        filter_v = self.filter(v_cor)
        # ops.print_("filter_u",filter_u.shape)
        # filter_u = self.filter(u_prev)
        # filter_v = self.filter(v_prev)
        # time1 = time.time()

        u_res =(self.vis * (filter_u[:, 2:3, :, :] + filter_u[:, 3:4, :, :]) - (
                u_prev * filter_u[:, 0:1, :, :] + v_prev * filter_u[:, 1:2, :, :]))

        v_res = (self.vis * (filter_v[:, 2:3, :, :] + filter_v[:, 3:4, :, :]) - (
                u_prev * filter_v[:, 0:1, :, :] + v_prev * filter_v[:, 1:2, :, :]))
        # ops.print_("u_next", u_next.shape)
        # ops.print_("v_next", v_next.shape)
        return u_res, v_res
    def cal_cell(self, h):
        # u_prev = h[:, 0:1, ...]
        # v_prev = h[:, 1:2, ...]
        # # 1,1,25,25 - > 1,25,25,1   0,3,1,2
        #
        # u_prev_trans = ops.transpose(u_prev, (0, 2, 3, 1))
        # v_prev_trans = ops.transpose(v_prev, (0, 2, 3, 1))
        #
        # u_cor = self.u_cor(u_prev_trans)
        # v_cor = self.v_cor(v_prev_trans)
        # u_cor = ops.transpose(u_cor, (0, 3, 1, 2))
        # v_cor = ops.transpose(v_cor, (0, 3, 1, 2))
        #
        # filter_u = self.filter(u_cor)
        # filter_v = self.filter(v_cor)
        #
        #
        # u_next = u_prev + (self.vis * (filter_u[:, 2:3, :, :] + filter_u[:, 3:4, :, :]) - (
        #         u_prev * filter_u[:, 0:1, :, :] + v_prev * filter_u[:, 1:2, :, :])) * self.dt
        #
        # v_next = v_prev + (self.vis * (filter_v[:, 2:3, :, :] + filter_v[:, 3:4, :, :]) - (
        #         u_prev * filter_v[:, 0:1, :, :] + v_prev * filter_v[:, 1:2, :, :])) * self.dt
        #
        # ch = ops.concat((u_next, v_next), axis=1)
        u_prev = h[:, 0:1]
        v_prev = h[:, 1:2]
        k1_u, k1_v = self.rk(h)
        u1 = u_prev + k1_u * self.dt / 2.0
        v1 = v_prev + k1_v * self.dt / 2.0

        k2_u, k2_v = self.rk(ops.concat((u1, v1), axis=1))
        u2 = u_prev + k2_u * self.dt / 2.0
        v2 = v_prev + k2_v * self.dt / 2.0

        k3_u, k3_v = self.rk(ops.concat((u2, v2), axis=1))
        u3 = u_prev + k3_u * self.dt
        v3 = v_prev + k3_v * self.dt

        k4_u, k4_v = self.rk(ops.concat((u3, v3), axis=1))
        u_next = u_prev + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0 * self.dt
        v_next = v_prev + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0 * self.dt
        # u_next = u_prev + k1_u*self.dt
        # v_next = v_prev + k1_v*self.dt
        ch = ops.concat((u_next, v_next), axis=1)
        return ch

    def construct(self, init):
        # u0, v0 = init[:, 0:1, :, :].cuda(), init[:, 1:2, :, :].cuda()
        outputs_uv = []
        internel_uv = 0
        # name = "recurrent_cell"
        for step in range(self.steps - 1):
            if step == 0:

                uv = init
                internel_uv = uv

            uv = internel_uv
            uv = self.cal_cell(uv)   # 20,2,25,25#edit

            # print(step,torch.max(uv))
            internel_uv = uv
            if uv.shape[0] > 1:
                # outputs_uv.append(internel_uv.unsqueeze(dim=0))
                outputs_uv.append(expand_dims(internel_uv,0))
            else:  # infer
                outputs_uv.append(internel_uv)

        # if batch_id % 20 == 0:
        #     print("recurrent cost",time2-time1)
        outputs_uv = mindspore.ops.concat(tuple(outputs_uv), axis=0)#step,batch,channel,x,y
        # outputs_uv = transpose(outputs_uv,(1, 0, 2, 3, 4))
        return outputs_uv


class Filter(nn.Cell):
    def __init__(self):
        super(Filter, self).__init__()

        self.dx = Conv(order=1)
        # self.dx = nn.Conv2d(1,1,5,stride=1,padding=0)
        # self.dy = self.dx
        self.dxx = Conv(order=2)
        # self.dxx = nn.Conv2d(1,1,5,stride=1,padding=0)
        # self.dyy = self.dxx

    def padMethod(self, u_cor):
        # periodic padding
        u_cor_pad = ops.concat((u_cor[:, :, :, -2:], u_cor, u_cor[:, :, :, 0:2]), axis=3)
        u_cor_pad = ops.concat((u_cor_pad[:, :, -2:, :], u_cor_pad, u_cor_pad[:, :, 0:2, :]), axis=2)
        return u_cor_pad

    def construct(self, x):
        transpose = mindspore.ops.Transpose()
        x = self.padMethod(x)
        dx = self.dx(x)
        dy = transpose(self.dx(transpose(x, (0, 1, 3, 2))), (0, 1, 3, 2))
        # res = self.dx(x.permute(0,1,3,2)).permute(0,1,3,2) - dy
        dxx = self.dxx(x)
        dyy = transpose(self.dxx(transpose(x, (0, 1, 3, 2))), (0, 1, 3, 2))
        res = mindspore.ops.concat((dx, dy, dxx, dyy), axis=1)
        return res


class Conv(nn.Cell):
    def __init__(self, order):
        super(Conv, self).__init__()
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = 5
        self.deno = deno
        self.order = order
        self.conv2d = ops.Conv2D(out_channel=1, kernel_size=5)
        # conv2d = ops.Conv2D(out_channel=32, kernel_size=3)
        if self.order == 2:
            self.deno = self.deno ** 2
        # self.padding = int(self.kernel_size-1) / 2

        # self.matrix = zeros((5, 5), mindspore.float32)

        self.matrix_3 = mindspore.Parameter(Tensor(np.random.randn(3, 3), dtype=mindspore.dtype.float32),
                                            requires_grad=True)

    def get_kernel(self):
        matrix = zeros((5, 5), mindspore.float32)
        matrix[0, 0] = self.matrix_3[0, 0]
        matrix[0, 1] = self.matrix_3[0, 1]
        matrix[1, 0] = self.matrix_3[1, 0]
        matrix[1, 1] = self.matrix_3[1, 1]
        matrix[0, 2] = self.matrix_3[0, 2]
        matrix[1, 2] = self.matrix_3[1, 2]
        matrix[2, 0] = self.matrix_3[2, 0]
        matrix[2, 1] = self.matrix_3[2, 1]

        if self.order == 1:
            # 1
            matrix[0, 3] = -matrix[0, 1]
            matrix[0, 4] = -matrix[0, 0]
            matrix[1, 3] = -matrix[1, 1]
            matrix[1, 4] = -matrix[1, 0]
            # 2
            matrix[3, 0] = -matrix[1, 0]
            matrix[4, 0] = -matrix[0, 0]
            matrix[3, 1] = -matrix[1, 1]
            matrix[4, 1] = -matrix[0, 1]
            # 3
            matrix[3, 3] = -matrix[1, 3]
            matrix[4, 3] = -matrix[0, 3]
            matrix[3, 4] = -matrix[1, 4]
            matrix[4, 4] = -matrix[0, 4]
            # middle
            matrix[3, 2], matrix[4, 2] = -matrix[1, 2], -matrix[0, 2]
            matrix[2, 3], matrix[2, 4] = -matrix[2, 1], -matrix[2, 0]
            # temp = matrix[2,2]
            # matrix[2,2] = -(sum(matrix) - temp)
            matrix[2, 2] = 0
        else:
            matrix[0, 3] = matrix[0, 1]
            matrix[0, 4] = matrix[0, 0]
            matrix[1, 3] = matrix[1, 1]
            matrix[1, 4] = matrix[1, 0]
            # 2
            matrix[3, 0] = matrix[1, 0]
            matrix[4, 0] = matrix[0, 0]
            matrix[3, 1] = matrix[1, 1]
            matrix[4, 1] = matrix[0, 1]
            # 3
            matrix[3, 3] = matrix[1, 3]
            matrix[4, 3] = matrix[0, 3]
            matrix[3, 4] = matrix[1, 4]
            matrix[4, 4] = matrix[0, 4]
            # middle
            matrix[3, 2], matrix[4, 2] = matrix[1, 2], matrix[0, 2]
            matrix[2, 3], matrix[2, 4] = matrix[2, 1], matrix[2, 0]
            # temp = matrix[2,2]
            # matrix[2,2] = -(sum(matrix) - temp)
            matrix[2, 2] = -(
                    (matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1]) * 4 + (
                    matrix[0, 2] + matrix[1, 2] + matrix[2, 0] + matrix[2, 1]) * 2)

        return matrix

    def construct(self, x):
        # update matrix
        # whether update matrix
        m = self.get_kernel()

        # print("m is",type(m))
        # print("m.value",m.value())
        # print("self.matrix.value()",self.matrix.value())
        # update filter weight
        # with torch.no_grad():
        # weight = self.matrix.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = expand_dims(expand_dims(m, 0), 0)

        # print(weight)
        # print("*************")
        # prints(weight.shape)
        # time1 = time.time()
        # central = mindspore.ops.Conv2D(x, weight=matrix.cuda(), stride=1, padding=0).cuda() /self.deno
        central = self.conv2d(x, weight) / self.deno

        # x = mindspore.ops.Conv2D(u_pad_2, weight=mindspore.Tensor(dx_2d_op).cuda(), stride=1, padding=0).cuda()
        # time2 = time.time()
        # print(time2-time1)
        return central


def generate_dataset(data, train_win, icnum, nolap=True):  # 3,400,2,64,64  -> 120 10 2 64 64
    gap = 8# degree of overlap  if nolap=False
    if nolap:
        gap = train_win
    start = [i for i in range(0, len(data[0]) - train_win + 1, gap)]
    random.shuffle(start)
    train_set = []
    for i in start:
        for j in range(icnum):
            train_set.append(data[j][i:i + train_win])
    all_shuffle = []
    rc = [i for i in range(len(train_set))]
    random.shuffle(rc)
    for i in rc:
        all_shuffle.append(train_set[i])
    all_shuffle = np.array(all_shuffle)
    # train_set = torch.cat(tuple(train_set),dim=0)
    return all_shuffle


def plot_loss(name, loss):
    data_loss = loss
    # for i in range(len(loss)):
    #     data_loss.append(loss[i][0])
    iter = [i for i in range(1, len(loss) + 1)]
    plt.plot(iter, data_loss, color="red", label=name + "_loss")
    plt.title("Loss_iters_" + name_conf + "_" + name, fontsize=24)
    plt.xlabel("iters", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.tick_params(axis="both", labelsize=14)
    plt.savefig("../figure3/" + name_conf + "_" + name + ".png")
    plt.legend(fontsize=16)
    plt.show()
    plt.close()
    print(name + "plot over")


def load_model(model):
    print("load model....")
    param_dict = mindspore.load_checkpoint('./model/checkpoint' + name_conf + '.ckpt')
    # checkpoint = mindspore.load_checkpoint('../model/checkpoint3_2_improve.pt')
    mindspore.load_param_into_net(model, param_dict)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(0.0))
    # best_loss = checkpoint['best_loss']
    # print("best_loss:", best_loss)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler = mindspore.nn.piecewise_constant_lr(optimizer, step_size=100, gamma=0.98)
    # return model, optimizer, scheduler, best_loss
    return model


def infer2(model, init, infersteps):
    ou = []
    outputs_uv = []
    model.steps = infersteps
    outputs_uv = model(init)
    return outputs_uv


def forward_fn(inputs, targets):
    output_uv = model(inputs)
    logits = transpose(output_uv, (1, 0, 2, 3, 4))
    loss = mse_loss(logits, targets)
    return loss


def train_step(grad_fn, inputs, targets, optimizer):
    loss, grads = grad_fn(inputs, targets)  # get values and gradients
    loss = ops.depend(loss, optimizer(grads))
    return loss


def train_dl(model, train_loader, n_iters, learning_rate, train_window, batch_size, cont=True):
    if cont:
        model, optimizer, scheduler, best_loss = load_model(model)
        if pretrain:
            best_loss = 0.0001
    else:
        optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
        # milestone = [1, 2, 10, 20]
        # learning_rates = [0.01, 0.005, 0.001, 0.0001]
        # scheduler = nn.piecewise_constant_lr(optimizer, step_size=100, gamm?a=0.96)
        # scheduler = nn.piecewise_constant_lr(milestone, learning_rates)
        best_loss = 100000

    # torch.autograd.set_detect_anomaly = True

    # mse_loss = nn.MSELoss()

    epoch_loss_list = []
    print("start training...")
    model.steps = train_window
    grad_fn = ops.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=False)

    for epoch in range(n_iters):

        # print(torch.cuda.memory_summary())requires_grad
        time1 = time.time()
        loss_data, loss_phy, loss, batch_loss, data_loss, phy_loss = [0] * 6
        # num_time_batch = (len(truth) - 1) // batch_size
        epoch_loss = 0
        batch_id = 0
        # print(model.filter.dx.matrix.data())
        for batch_data in train_loader.create_dict_iterator():
            # print(batch_data.shape)
            batch_id += 1
            sample_init = Tensor(batch_data['data'].squeeze())
            # print("sample_init.shape",sample_init.shape)
            gt_uv = Tensor(batch_data['label'])
            loss = train_step(grad_fn,sample_init, gt_uv,optimizer)
            # loss = forward_fn(sample_init, gt_uv)
            # print(loss)
            # loss,grad = grad_fn(sample_init,gt_uv)
            # loss = ops.depend(loss, optimizer(grad))

            # print(len(batch_data))

            # if batch_id % 5 == 0:
            # print('[%d/%d %d%% batch%d] cur_batch_loss(avg): %.11f' % (
            #         (epoch + 1), n_iters, ((epoch + 1) / (n_iters) * 100.0), batch_id, loss))
            # print(model.filter.dx.weight.data[0])
            # print("batch_cost:", batch_time2 - batch_time1)
            # if loss.item() < best_loss:
            if loss < best_loss:
                best_loss = loss
                print('save model !!!')
                # print("cur data loss", data_loss.item())
                mindspore.save_checkpoint(model, "./model/checkpoint" + name_conf + ".ckpt")
            epoch_loss += loss
        # print("epoch loss: ",epoch_loss)
        epochloss = epoch_loss / batch_id
        # print(epochloss)
        epoch_loss_list.append(np.float64(epochloss))
        print('-------- At %d epoch ------------' % (epoch + 1))
        print('[%d/%d %d%%] loss: %.11f' % (
            (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), epochloss / len(batch_data)))
        time2 = time.time()
        print("epoch cost", time2 - time1)
    return epoch_loss_list


def train_dl2(model, train_loader, n_iters, learning_rate, train_window, batch_size, cont=True):
    if cont:
        model, optimizer, scheduler, best_loss = load_model(model)
        if pretrain:
            best_loss = 0.0001
    else:
        optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate, weight_decay=0.9)
        # milestone = [1, 2, 10, 20]
        # learning_rates = [0.01, 0.005, 0.001, 0.0001]
        # scheduler = nn.piecewise_constant_lr(optimizer, step_size=100, gamm?a=0.96)
        # scheduler = nn.piecewise_constant_lr(milestone, learning_rates)
        best_loss = 100000

    # torch.autograd.set_detect_anomaly = True

    mse_loss = nn.MSELoss()

    epoch_loss_list = []
    print("start training...")
    model.steps = train_window
    for epoch in range(n_iters):
        # print(torch.cuda.memory_summary())
        time1 = time.time()
        loss_data, loss_phy, loss, batch_loss, data_loss, phy_loss = [0] * 6
        # num_time_batch = (len(truth) - 1) // batch_size
        epoch_loss = 0
        batch_id = 0
        for batch_data in train_loader.create_dict_iterator():
            batch_id += 1
            sample_init = Tensor(batch_data['data'].squeeze())
            gt_uv = Tensor(batch_data['label'])

            # optimizer.zero_grad()
            # output_uv = model(sample_init.cuda())
            output_uv = model(sample_init, 0)
            output_uv = mindspore.ops.concat(tuple(output_uv), axis=0)
            # output_uv = output_uv.permute(1, 0, 2, 3, 4)
            output_uv = transpose(output_uv, (1, 0, 2, 3, 4))

            # cha = len(gt_uv) - len(output_uv)
            # print(cha)
            # if cha != 0:
            #     gt_uv = gt_uv[:cha]
            # phy_uv = output_uv
            # data_loss = mse_loss(output_uv, gt_uv.cuda())  # data loss
            data_loss = mse_loss(output_uv, gt_uv)  # data loss

            data_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            if batch_id % 5 == 0:
                print('[%d/%d %d%% batch%d] cur_batch_loss(avg): %.11f' % (
                    (epoch + 1), n_iters, ((epoch + 1) / (n_iters) * 100.0), batch_id, data_loss.item()))
            # print("batch_cost:", batch_time2 - batch_time1)
            if data_loss.item() < best_loss:
                best_loss = data_loss.item()
                print('save model !!!')
                # print("cur data loss", data_loss.item())
                mindspore.save_checkpoint({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, "./model/checkpoint" + name_conf + ".ckpt")
            epoch_loss += data_loss.item()
        epoch_loss_list.append(epoch_loss)
        print('-------- At %d epoch ------------' % (epoch + 1))
        # print('[%d/%d %d%%] loss: %.11f, data_loss:%.11f, phy_loss:%.7f' % (
        #     (epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), batch_loss / len(batch_data[0]), batch_loss / len(batch_data[0]), phy_loss))
        time2 = time.time()
        print("epoch cost", time2 - time1)
    return epoch_loss_list


def cal_reffe(output, truth):
    # output = mindspore.Tensor(output)
    # output = output.clone().detach()
    output = output
    nume = mindspore.ops.LpNorm(output - truth)
    deno = mindspore.ops.LpNorm(truth)
    epsino = nume / deno
    return epsino

def plot_err_prop(name_conf,UV_TRUTH,UV_PINN,dt,fig_save_dir):
    # t = np.linspace(0, dt, 2001)

    UV_TRUTH = np.array(UV_TRUTH)
    UV_PINN = np.array(UV_PINN)

    # rc('text', usetex=True)
    # rc('legend', fontsize=14)
    # rc('font', family='Serif')

    # Accumulative error (accumulative) - used in paper
    eps = 1e-4

    MSE_PINN = np.mean((UV_TRUTH - UV_PINN) ** 2, axis=(0, 2, 3))
    accum_PINN = np.array([[i * dt, eps + np.sqrt(MSE_PINN[:i + 1].mean())] for i in range(0, UV_PINN.shape[1], 1)])

    # plot figures
    # fig, ax = plt.subplots(figsize=(6.5, 4))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0.12, 0.15, 0.85, 0.80])  # [left, bottom, width, height]
    # ax.fill_between(accum_PINN[:501, 0], 10 + 0 * accum_PINN[:501, 1], 0 * accum_PINN[:501, 1],
    #                 facecolor='lightskyblue',
    #                 alpha=0.1, )  # hatch='///', edgecolor='red'
    # ax.fill_between(accum_PINN[500:, 0], 10 + 0 * accum_PINN[500:, 1], 0 * accum_PINN[500:, 1], facecolor='orange',
    #                 alpha=0.1, )  # hatch='///', edgecolor='blue'
    #    # plot PINN
    # ax.plot(accum_PINN[:, 0], accum_PINN[:, 1], alpha=0.8, linewidth=2, label=r'$\mathrm{ours}$', color='black')
    ax.plot(accum_PINN[:, 0], accum_PINN[:, 1], alpha=0.8, linewidth=2, color='black')

    # ax.set_xlim([0, 1.4])
    ax.set_ylim([eps, 1e-1])
    # ax.set_xticks([i for i in range(0,21)])
    # ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,1.4])
    ax.set_yticks([eps, 1e-1])
    ax.set_yscale('log')
    ax.set_ylabel(r'a-RMSE', fontsize=14)
    # ax.set_xlabel(r'$t$', fontsize=14, labelpad=-0.0)
    ax.set_xlabel('t', fontsize=14, labelpad=-0.0)
    ax.tick_params(labelsize=14, direction='in')
    # plt.legend(frameon=False, loc='upper left')
    plt.legend(frameon=False, loc='upper left')
    # plt.text(0.75, 0.0005, r'$\mathrm{Training}$', fontsize=18, color='steelblue')
    # plt.text(2.5, 0.0005, r'$\mathrm{Extrapolation}$', fontsize=18, color='orange')

    # plt.text(0.75, 0.0005, r'Training', fontsize=18, color='steelblue')
    # plt.text(2.5, 0.0005, r'Extrapolation', fontsize=18, color='orange')

    plt.savefig(fig_save_dir + name_conf+'burgers-err-propa.png')
    # plt.savefig(fig_save_dir + 'ablation-burgers.png')
    plt.show()
def plot_err_prop(name_conf,UV_TRUTH,UV_Net,dt,fig_save_dir):
    # t = np.linspace(0, dt, 2001)

    UV_TRUTH = np.array(UV_TRUTH, dtype=float)
    UV_Net = np.array(UV_Net, dtype=float)

    # rc('text', usetex=True)
    # rc('legend', fontsize=14)
    # rc('font', family='Serif')

    # Accumulative error (accumulative) - used in paper
    eps = 1e-4

    MSE_PINN = np.mean((UV_TRUTH - UV_Net) ** 2, axis=(0, 2, 3))
    accum_PINN = np.array([[i * dt, eps + np.sqrt(MSE_PINN[:i + 1].mean())] for i in range(0, UV_Net.shape[1], 1)])

    # plot figures
    # fig, ax = plt.subplots(figsize=(6.5, 4))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0.12, 0.15, 0.85, 0.80])  # [left, bottom, width, height]
    # ax.fill_between(accum_PINN[:501, 0], 10 + 0 * accum_PINN[:501, 1], 0 * accum_PINN[:501, 1],
    #                 facecolor='lightskyblue',
    #                 alpha=0.1, )  # hatch='///', edgecolor='red'
    # ax.fill_between(accum_PINN[500:, 0], 10 + 0 * accum_PINN[500:, 1], 0 * accum_PINN[500:, 1], facecolor='orange',
    #                 alpha=0.1, )  # hatch='///', edgecolor='blue'
    #    # plot PINN
    # ax.plot(accum_PINN[:, 0], accum_PINN[:, 1], alpha=0.8, linewidth=2, label=r'$\mathrm{ours}$', color='black')
    ax.plot(accum_PINN[:, 0], accum_PINN[:, 1], alpha=0.8, linewidth=2, color='black')

    ax.set_xlim([0, 1.4])
    ax.set_ylim([eps, 1e-1])
    # ax.set_xticks([i for i in range(0,21)])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,1.4])
    ax.set_yticks([eps, 1e-1])
    ax.set_yscale('log')
    ax.set_ylabel(r'a-RMSE', fontsize=14)
    # ax.set_xlabel(r'$t$', fontsize=14, labelpad=-0.0)
    ax.set_xlabel('t', fontsize=14, labelpad=-0.0)
    ax.tick_params(labelsize=14, direction='in')
    # plt.legend(frameon=False, loc='upper left')
    plt.legend(frameon=False, loc='upper left')
    # plt.text(0.75, 0.0005, r'$\mathrm{Training}$', fontsize=18, color='steelblue')
    # plt.text(2.5, 0.0005, r'$\mathrm{Extrapolation}$', fontsize=18, color='orange')

    # plt.text(0.75, 0.0005, r'Training', fontsize=18, color='steelblue')
    # plt.text(2.5, 0.0005, r'Extrapolation', fontsize=18, color='orange')

    plt.savefig(fig_save_dir + name_conf+'burgers-err-propa.png')
    # plt.savefig(fig_save_dir + 'ablation-burgers.png')
    plt.show()
def cal_error4onedata(truth,i, model, ifplot=False, ploterrorprop=True, names="pred", rate=10):
    # init_u = truth[0:1].cuda()
    init_u = Tensor(truth[0:1])
    truth = truth[1:]
    # output = infer2(model, init_u.cuda(), inferstep).cpu().detach()
    output = infer2(model, init_u, inferstep)#.cpu().detach()
    print("output.shape", output.shape)
    print("truth.shape", truth.shape)

    if ploterrorprop:
        print("error propagation ploting")
        UV_TRUTH = mindspore.Tensor(truth).permute(1, 0, 2, 3)
        UV_PINN = mindspore.Tensor(output).permute(1, 0, 2, 3)
        err_save_dir = "./figure3/"
        plot_err_prop(name_conf + "training"+str(i), UV_TRUTH, UV_PINN, delta_t, err_save_dir)
    # output_u = output[:, 0:1, :, :]
    # output_v = output[:, 1:2, :, :]
    name = name_conf + names
    mse_loss = ops.mse_loss(mindspore.Tensor(truth), mindspore.Tensor(output))
    print("mse loss: ", mse_loss)
    # if ifplot:
    #     plotgif2(name + "u", output[:, 0:1, ...], rate)
    #     plotgif2(name + "v", output[:, 1:2, ...], rate)
    # relative_fferror_u = cal_reffe(output[:, 0:1, ...], truth[:, 0:1, ...])
    # relative_fferror_v = cal_reffe(output[:, 1:2, ...], truth[:, 1:2, ...])
    # print("relative_full_field l2 error(testing)")
    # print("u: ", relative_fferror_u)
    # print("v: ", relative_fferror_v)
    # # ===start===plot_err_prop_curve-----
    # UV_TRUTH = truth.permute(1, 0, 2, 3)
    # UV_PINN = mindspore.Tensor(output[:, :2, ...]).permute(1, 0, 2, 3)
    # err_save_dir = "./figure3/"
    # plot_err_prop(name_conf + "test", UV_TRUTH, UV_PINN, delta_t, err_save_dir)
    # return relative_fferror_u, relative_fferror_v


def get_loader(num, train_win, batch_size, nolap=True):
    all_data = []
    for i in range(1, num + 1):
        # cur_UV = sio.loadmat('./data/Burgers_2101x2x100x100_[RK4,R=200,dt=0_001,#seed' + str(i) + '].mat')['uv']
        # cur_UV = np.load('./data/Burgers_2101x2x100x100_[RK4,R=200,dt=0_001,#seed' + str(i) + '].npy')
        cur_UV = np.load('./data104/Burgers_2101x2x104x104_[RK4,R=200,dt=0_001,#seed' + str(i) + '].npy')
        # np.save('./data/Burgers_2101x2x100x100_[RK4,R=200,dt=0_001,#seed' + str(i) + '].npy',cur_UV)
        # cur_uv = Tensor.from_numpy(cur_UV[100:2101, ...]).float()
        cur_uv = np.ascontiguousarray(cur_UV[100:2101, ...], dtype=np.float32)
        cur_truth = cur_uv[:timesteps, :, ::down, ::down]
        # print(cur_truth.shape)
        all_data.append(cur_truth)
    all_data = np.array(all_data)
    all_data_set = generate_dataset(all_data, train_win=train_win, icnum=1, nolap=True)

    mydataset = MyDataset(all_data_set)

    dataset = ds.GeneratorDataset(mydataset, ['data', 'label'], shuffle=False)
    cur_loader = dataset.batch(batch_size, True)
    # for data in cur_loader.create_dict_iterator():
    #     data1 = data['data'].asnumpy()
    #     label1 = data['label'].asnumpy()

    # cur_loader = DataLoader(mydataset, batch_size, shuffle=True)
    return cur_loader
def loaddataIC2(num, time_down, down, time_steps):#
    # path = "./data/"
    path = "/mnt/DEV_A/wangqi/data/2dburgers/"
    data_all = []
    if trainStage:
        start = 1
    else:
        start = 11
    end = start + num
    for i in range(start,end):

        cur_UV =np.load('./data/Burgers_2101x2x104x104_[RK4,R=200,dt=0_001,#seed' + str(i) + '].npy')
        cur_data = np.ascontiguousarray(cur_UV[100:2101, ...], dtype=np.float32)

        cur_data = cur_data[::time_down, :,::down,::down]
        cur_data = cur_data[:time_steps, ...]
        print(cur_data.shape)
        print("curdata.shape",cur_data.shape)
        data_all.append(cur_data)

    return np.array(data_all)
def saveEveryIC():

    data = loaddataIC2(5, time_down, down, timesteps)
    print("time_down",time_down)
    print("training data.shape", data.shape)
    train_win = 32
    flag = "MindSymBurgers_"+"float32"
    for i in range(len(data)):
        cur_data = data[i:i+1]
        print("cur data.shape", cur_data.shape)
        cur_data_set = generate_dataset(cur_data, train_win=train_win, icnum=1, nolap=True)
        print("cur_IC_sample_set.shape", cur_data_set.shape)
        cur_nameconf = str(train_win)+"_"+str(cur_data_set.shape[0]) + "x" + str(cur_data_set.shape[1]) + "x" + str(
            cur_data_set.shape[2]) + "x" + str(cur_data_set.shape[3]) + "x" + str(
            cur_data_set.shape[4])
        np.save("./data/sample/"+flag+"group"+str(i+1)+"_" + cur_nameconf + "_uv_shufflesample.npy", cur_data_set)
        print("save at: "+"./data/sample/"+flag+"group"+str(i+1)+"_" + cur_nameconf + "_uv_shufflesample.npy")
def selectSample(num,everynum):
    all_sample = []
    for i in range(1,num+1):

        cur_ = np.ascontiguousarray(np.load("./data/sample/MindSymBurgers_float32group"+str(i)+"_32_40x32x2x26x26_uv_shufflesample.npy"),dtype=np.float32)
        # cur_res = cur_[mindspore.ops.randperm(len(cur_))]
        cur_res = cur_
        cur_res = cur_res[:everynum]
        all_sample.extend(cur_res)
    all_sample = np.array(all_sample)
    # all_sample_shuffle = all_sample[ops.randperm(len(all_sample))]
    all_sample_shuffle = all_sample
    print("all_sample.shape",all_sample.shape)


    return all_sample_shuffle
if __name__ == '__main__':

    timesteps = 1280
    down = 4
    pretrain_iters = 20
    pretrain_window = 5
    train_window = 32
    time_down = 1
    batch_size = 40
    deno = 1 / 104.0 * 4
    modes = 12
    width = 12
    print("modes: ", modes)
    print("width: ", width)
    #inferstep = 1400
    inferstep=32
    resolution = 26
    name_conf = "test_mindspore"
    n_iters = 1000
    # dt = 0.00025
    delta_t = 0.001
    vis = 1 / 200.0
    learning_rate = 0.01
    num = 5
    trainStage = False
    # saveEveryIC()#./data/sample/MindSymBurgers_float32group5_32_40x32x2x26x26_uv_shufflesample.npy
    # exit()
    pretrain_loader = get_loader(num, 5, batch_size=40, nolap=True)
    train_loader = get_loader(num, train_window, batch_size, nolap=True)
    global_epoch = 0
    model = RCNN(train_window)

    testStage = True
    ploterrorprop = True
    if trainStage:
        print("*****************trainingStage****************")
        all_sample = selectSample(5, 200)
        for end_sample in [2000]:
            # group += 1
            #
            # if group == 1:
            #     n_iters = 2000

            cur_feedsample = all_sample[:end_sample]
            # cur_feedsample_shuffle = cur_feedsample[ops.randperm(len(cur_feedsample))]
            cur_feedsample_shuffle = cur_feedsample
            print("cur_feedsample",cur_feedsample_shuffle.shape)
            mydataset = MyDatasets(cur_feedsample_shuffle)

            dataset = ds.GeneratorDataset(mydataset, ['data', 'label'], shuffle=True)
            train_loader = dataset.batch(batch_size, True)
            # val_loader = train_loader#fake
            loss = train_dl(model, train_loader, n_iters, learning_rate, train_window, batch_size, cont=False)

    if testStage:
        print("*****************testingStage****************")
        # model, optimizer, scheduler, _ = load_model(model)
        model = load_model(model)
        path = "./data104/"
        ul, vl = 0, 0
        #num = 10
        num=1
        #for i in range(12, num + 12):  # 1-10train 11-20test
        for i in range(1, num+1):
            print(i)
            # cur_UV = sio.loadmat('./data/Burgers_2101x2x100x100_[RK4,R=200,dt=0_001,#seed' + str(i) + '].mat')['uv']
            cur_UV = np.load('./data104/Burgers_2101x2x104x104_[RK4,R=200,dt=0_001,#seed' + str(i) + '].npy')
            curuv = np.ascontiguousarray(cur_UV[100:2101, ...], dtype=np.float32)[:inferstep, :, ::4, ::4]
            cal_error4onedata(curuv,i, model, ploterrorprop=ploterrorprop)
