
import numpy as np
import os
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
import scipy.io as io
import torch
import torch.nn.functional as F
import h5py
from libtiff import TIFF


# import torchvision.models as models

def Theta2TransMatrix(param):
    theta = param.theta
    center = param.center
    ShiftMatrix_1 = np.matrix([[1, 0, 0],
                              [0, 1, 0],
                              [-center[0], -center[1], 1]])
    RotationMatrix_f = np.matrix([[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])
    RotationMatrix_b = np.matrix([[math.cos(-theta), math.sin(-theta), 0],
                                [-math.sin(-theta), math.cos(-theta), 0],
                                [0, 0, 1]])
    ShiftMatrix_2 = np.matrix([[1, 0, 0],
                             [0, 1, 0],
                             [center[0], center[1], 1]])
    param.TransMatrix_0 = torch.from_numpy(RotationMatrix_f).float().to(param.device)
    param.TransMatrix_1 = torch.from_numpy(ShiftMatrix_1 * RotationMatrix_f * ShiftMatrix_2).float().to(param.device)
    param.TransMatrix_2 = torch.from_numpy(RotationMatrix_b).float().to(param.device)

    return param


def Rotation_XZ(X, param):
    X_coor = torch.linspace(-1, 1, param.PhantomSize[0]).to(param.device)
    Z_coor = torch.linspace(-1, 1, param.PhantomSize[2]).to(param.device)

    [X_coor, Z_coor] = torch.meshgrid(X_coor, Z_coor)

    temp_ones = torch.ones([param.PhantomSize[0], param.PhantomSize[2]]).to(param.device)

    coor = torch.stack([X_coor, Z_coor, temp_ones], dim=2)

    coor = coor.reshape([param.PhantomSize[0]*param.PhantomSize[2], 3])
    coor = coor.mm(param.TransMatrix_0)
    coor = coor.reshape([param.PhantomSize[0], param.PhantomSize[2], 3])

    coor = coor[:, :, 0:2].reshape(1, param.PhantomSize[0], param.PhantomSize[2], 2)
    X_rotate = X * 0

    for i_Layer in range(0, param.PhantomSize[1]):
        X_layer = X[i_Layer, :, :].squeeze().reshape(1, 1, param.PhantomSize[0], param.PhantomSize[2])
        # plt.imshow(X_layer.cpu().detach().numpy())
        X_rotate[i_Layer, :, :] = F.grid_sample(X_layer, coor,
                                                mode='bilinear',
                                                padding_mode='zeros',
                                                align_corners=True).squeeze()
    X_rotate = X_rotate.permute([0, 2, 1])
    return X_rotate


class Model_Projection(torch.nn.Module):

    def __init__(self, Phantom, param, PSF):
        super().__init__()
        self.Phantom = torch.nn.Parameter(Phantom, requires_grad=param.FlagGrad_Phantom)
        self.param = param
        self.PSF = PSF
    def forward(self):
        PhantomSize = self.param.PhantomSize
        PSFSize = self.param.PSFSize
        self.param = Theta2TransMatrix(self.param)
        Phantom_rot = Rotation_XZ(self.Phantom, self.param)
        Phantom_fft = torch.rfft(Phantom_rot, 3, onesided=False)
        ProjAngle = self.param.ProjAngle
        PSF_SingleAngle = self.PSF[:, :, ProjAngle[0], ProjAngle[1], :]
        # plt.imshow(np.squeeze(Phantom[:, :, 100].cpu().detach().numpy()))
        pd_temp_1 = int((PhantomSize[0] - PSFSize[0]) / 2)
        pd_temp_2 = int((PhantomSize[1] - PSFSize[1]) / 2)
        pd_temp = (0, 0, pd_temp_2, pd_temp_2, pd_temp_1, pd_temp_1)
        PSF_SingleAngle_enlarge = F.pad(PSF_SingleAngle, pd_temp, "constant", 0)
        temp = sum(sum(sum(PSF_SingleAngle_enlarge, 0), 0), 0)
        PSF_SingleAngle_enlarge = PSF_SingleAngle_enlarge/temp
        PSF_SingleAngle_enlarge = torch.flip(PSF_SingleAngle_enlarge, [2])
        del PSF_SingleAngle

        PSF_SingleAngle_enlarge_fft = torch.rfft(PSF_SingleAngle_enlarge, 3, onesided=False)

        conv_slice_fft_r = PSF_SingleAngle_enlarge_fft[:, :, :, 0] * Phantom_fft[:, :, :, 0] -\
                         PSF_SingleAngle_enlarge_fft[:, :, :, 1] * Phantom_fft[:, :, :, 1]

        conv_slice_fft_i = PSF_SingleAngle_enlarge_fft[:, :, :, 0] * Phantom_fft[:, :, :, 1] +\
                           PSF_SingleAngle_enlarge_fft[:, :, :, 1] * Phantom_fft[:, :, :, 0]

        conv_fft = torch.stack([conv_slice_fft_r, conv_slice_fft_i], 3)

        conv2_fft = torch.sum(conv_fft, 2)

        conv2_unshift = torch.irfft(conv2_fft, 2, onesided=False, signal_sizes=self.Phantom.shape[0:2])

        shift_temp = int((525 + 1) / 2)
        conv_ZP = torch.roll(conv2_unshift, shifts=(shift_temp, shift_temp), dims=(0, 1))

        conv_ZP[torch.isnan(conv_ZP)] = 0
        conv_ZP[conv_ZP < 1e-8] = 0

        return conv_ZP


class parameter:
    def __init__(self):
        # Phantom
        self.PhantomSize = []

        # PSF
        self.PSFSize = []

        # device
        self.device = []


def Initialize_Refraction_ball(Num_RIGuassKernel, A_base, Sig_base):

    sig = Sig_base * np.ones([Num_RIGuassKernel, Num_RIGuassKernel, Num_RIGuassKernel])
    sig_swap = sig

    Xc = np.linspace(0, 1, Num_RIGuassKernel, dtype='float32')
    Yc = np.linspace(0, 1, Num_RIGuassKernel, dtype='float32')
    Zc = np.linspace(0, 1, Num_RIGuassKernel, dtype='float32')
    Yc, Xc, Zc = np.meshgrid(Xc, Yc, Zc)

    A = A_base[0] * np.ones([Num_RIGuassKernel, Num_RIGuassKernel, Num_RIGuassKernel])
    temp = (Xc-0.4)**2 + (Yc-0.5)**2 + 2*(Zc-0.5)**2 < 0.3**2
    A[temp] = A_base[1]

    # io.savemat('asig', {'sig': sig, 'A': A})

    A_swap = np.swapaxes(A, 0, 1)

    A = A_swap.flatten()  # [y, x, z]
    sig = sig_swap.flatten()  # [y, x, z]

    RefractionIndex = np.stack([A, sig], axis=1)

    return RefractionIndex


def Initialize_Refraction(Num_RIGuassKernel, A_base, Sig_base):

    sig = Sig_base * np.ones([Num_RIGuassKernel, Num_RIGuassKernel, Num_RIGuassKernel])
    sig_swap = sig

    A = A_base[0] * np.ones([Num_RIGuassKernel, Num_RIGuassKernel, Num_RIGuassKernel])

    # io.savemat('asig', {'sig': sig, 'A': A})

    A_swap = np.swapaxes(A, 0, 1)

    A = A_swap.flatten()  # [y, x, z]
    sig = sig_swap.flatten()  # [y, x, z]

    RefractionIndex = np.stack([A, sig], axis=1)

    return RefractionIndex


def Initialize_Phantom(PhantomSize):

    Phantom = np.zeros([PhantomSize[0], PhantomSize[1], PhantomSize[2]])
    Phantom[0::10, :, :] = 1
    Phantom[:, 0::10, :] = 1
    Phantom[:, :, 0::10] = 1

    return Phantom


if __name__ == '__main__':

    ## Initialize_PhantomSize
    # PhantomSize = [525, 525, 525]
    # PhantomSize = [315, 315, 315]
    # PhantomSize = [105, 105, 105]

    # Phantom_numpy = Initialize_Phantom(PhantomSize)
    data = io.loadmat('.\Phantom\Phantom_real_525_2.mat')
    Phantom_numpy = data['X_true']


    PhantomSize = list(Phantom_numpy.shape)

    # Phantom_numpy = Phantom_numpy.transpose((2, 0, 1))
    # io.savemat('Phantom', {'Phantom': Phantom_numpy})
    Phantom = Phantom_numpy.astype('float32')
    temp = np.max(np.max(np.max(Phantom, 0), 0), 0)
    Phantom = Phantom / temp * 1e6
    Phantom = torch.from_numpy(Phantom)
    # Phantom_numpy = np.squeeze(Phantom.numpy())
    # io.savemat('Phantom', {'Phantom': Phantom_numpy})


    ## Initialize_PSF


    mat = h5py.File('.\PSF\psf_sim_RoundAp_301_301_15_15_525.mat','r')
    PSF_numpy = np.transpose(mat['psf'])
    # uvIdx_numpy = np.transpose(mat['uvIdx'])

    # Phantom_numpy = Phantom_numpy.transpose((2, 0, 1))
    # io.savemat('Phantom', {'Phantom': Phantom_numpy})

    PSF_numpy = PSF_numpy.astype('float32')
    PSF = torch.from_numpy(PSF_numpy)
    PSFSize = PSF_numpy.shape


    # uvIdx_numpy = uvIdx_numpy.astype('float32')
    # uvIdx = torch.from_numpy(uvIdx_numpy)

    # Phantom_numpy = np.squeeze(Phantom.numpy())
    # io.savemat('Phantom', {'Phantom': Phantom_numpy})


    # Projection map
    data = io.loadmat('.\WJM_Map\wjm_chyf15_map.mat')
    map = data['map']

    ## rotation
    # theta = math.pi/2
    center = [0.5, 0.5]

    ## Projection
    ProjAngle = 1

    ## Parameter
    param = parameter()

    param.map = map

    param.FlagGrad_RefractionIndex = False
    param.FlagGrad_Phantom = False
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param.device = device

    # param.theta = theta
    param.center = center

    param.PhantomSize = PhantomSize
    param.PSFSize = PSFSize

    Phantom_gpu = Phantom.to(param.device)
    model = Model_Projection(Phantom_gpu, param, PSF).to(param.device)

    ## Rotation
    RotAngle = 8
    for RotAngle in [8, 10]:
        SavePath = 'Projection_Fourier6_5_Phantom507_Rot' + str(RotAngle) + '/'
        Projrction_numpy = np.zeros([PhantomSize[0], PhantomSize[0], PSFSize[2] * PSFSize[3]], dtype=np.single)
        if not os.path.exists(SavePath): os.mkdir(SavePath)
        # for i in range(0, RotAngle):
        for i_RotAngle in range(0, RotAngle):
            for i_ProjAngle in range(0, PSFSize[2]):
                for j_ProjAngle in range(0, PSFSize[3]):

                    model.param.ProjAngle = [i_ProjAngle, j_ProjAngle]
                    ProjNum = i_ProjAngle * PSFSize[2] + j_ProjAngle

                    time_1 = time()
                    model.param.theta = math.pi * 2 * i_RotAngle / RotAngle
                    Temp_Projrction = model.forward()
                    print(time() - time_1)

                    Projrction_numpy[:, :, ProjNum] = np.squeeze(Temp_Projrction.cpu().detach().numpy())
                # warped_Phantom_numpy = np.squeeze(warped_Phantom.detach().numpy())

            Addr = SavePath + 'Projrction_' + str(i_RotAngle) + '_' + str(RotAngle)
            io.savemat(Addr + '.mat', {'a': 1, 'Projrction': Projrction_numpy})

            tif = TIFF.open(Addr + '.tif', mode='w')
            for i in range(0, Projrction_numpy.shape[2]):
                tif.write_image(np.squeeze(Projrction_numpy[:, :, i]))

            tif.close()


    a = 1
