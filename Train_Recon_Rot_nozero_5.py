import sys
import numpy as np
import os
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from torch.utils.tensorboard import SummaryWriter
import math
import scipy.io as io
import torch
import torch.nn.functional as F
import h5py
import ModelProjectionFourier3_7 as Modle_PJ
from libtiff import TIFF

class parameter:
    def __init__(self):
        # Phantom
        self.PhantomSize = []

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
    temp = (Xc - 0.4) ** 2 + (Yc - 0.5) ** 2 + 2 * (Zc - 0.5) ** 2 < 0.3 ** 2
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

## Parameter
param = parameter()


## GPU or CPU
param.FlagGrad_RefractionIndex = True
device = torch.device("cuda:0")
# device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
param.device = device
writer = SummaryWriter('runs/exp')

## Initialize_PhantomSize
PhantomSize = [345, 345, 345]
# Phantom_numpy = Initialize_Phantom(PhantomSize)
# data = io.loadmat('Phantom_real_525.mat')
Phantom_numpy = np.zeros(PhantomSize) + 10000
Phantom = Phantom_numpy.astype('float32')
Phantom = torch.from_numpy(Phantom)
Phantom = Phantom.to(param.device)

## Initialize_PSF
mat = h5py.File('PSF/psf_sim_RoundAp_301_301_15_15_525.mat', 'r')
PSF_numpy = np.transpose(mat['psf'])
PSF_numpy = PSF_numpy.astype('float32')

# PSFSizeX = 101
# PSFSizeX_1 = int((301-1)/2 - (PSFSizeX-1)/2)
# PSFSizeX_2 = int((301-1)/2 + (PSFSizeX-1)/2 + 1)
# PSF_numpy = PSF_numpy[PSFSizeX_1: PSFSizeX_2, PSFSizeX_1: PSFSizeX_2, :, :, :]

PSFSizeX = 345
PSFSizeX_1 = int((525 - 1) / 2 - (PSFSizeX - 1) / 2)
PSFSizeX_2 = int((525 - 1) / 2 + (PSFSizeX - 1) / 2 + 1)
PSF_numpy = PSF_numpy[:, :, :, :, PSFSizeX_1: PSFSizeX_2]

PSF = torch.from_numpy(PSF_numpy)
PSFSize = PSF_numpy.shape

# Projection map
data = io.loadmat('WJM_Map/wjm_chyf15_map.mat')
map = data['map']

## Rotation
Num_Rotaion = 24
center = [0.5, 0.5]
EdgeRate = 1.42
temp = np.floor(PhantomSize[0] / EdgeRate / 2)
ErrorRange = [(PhantomSize[0] + 1)/2 - temp, (PhantomSize[0] + 1)/2 + temp]
ErrorRange = torch.arange(ErrorRange[0], ErrorRange[1]+1).to(param.device).long()

## uploading
param.center = center

param.PhantomSize = PhantomSize
param.PSFSize = PSFSize


## Undate Flag
param.FlagGrad_Phantom = True

## Training Parameter
Model = Modle_PJ.Model_Projection(Phantom, param, PSF).to(param.device)
criterion = torch.nn.MSELoss().to(device)
# lr = 0.0003
lr = 300
optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# ErrorRange =

## threshold
threshold = 1e-4

## Set Path
SavePath = 'Train_Recon_Rot' + str(Num_Rotaion) + '_711_NoZero/'
if not os.path.exists(SavePath): os.mkdir(SavePath)

WignerPath = 'Wigner_Sim/Train_Recon_Rot' + str(Num_Rotaion) + '_345/'
# WignerPath = 'Wigner_Sim/Wigner_Sim3_PSFRound_24angles_Phantom_real_525_2/'


## Training
stop_epoch = 50
Loss_list = np.zeros(stop_epoch*Num_Rotaion*map.shape[0])
for epoch in range(0, stop_epoch):
    # optimizer.param_groups[0]['lr'] = 0.00003
    # scheduler.step()  # 步长选择器更新
    Model.zero_grad()

    # for i_rotation in range(0, Num_Rotaion):
    for i_rotation in range(0, Num_Rotaion):
        Model.zero_grad()
        ## Loading Projrction
        data = io.loadmat(WignerPath + 'Projrction_' + str(i_rotation) + '_' + str(Num_Rotaion))
        Projrction = data['Projrction']
        Projrction = Projrction.astype('float32')
        # Projrction = Projrction.reshape(1, 1,
        #                                                 PhantomSize[0],
        #                                                 PhantomSize[1],
        #                                                 PSFSize[2]*PSFSize[3])
        Projrction = torch.from_numpy(Projrction)
        Projrction = Projrction.to(param.device)

        Model.train()  # 训练模式
        Model.param.theta = math.pi * 2 * i_rotation / Num_Rotaion

        ## Rotation
        # for i in range(0, RotAngle):
        total_loss = 0
        for i_Map in range(0, map.shape[0]):
            i_ProjAngle = map[i_Map, 0]
            j_ProjAngle = map[i_Map, 1]

            Model.zero_grad()
            Model.param.ProjAngle = [i_ProjAngle, j_ProjAngle]
            ProjNum = i_ProjAngle * PSFSize[2] + j_ProjAngle

            time_1 = time()
            Model.param.theta = math.pi * 2 * i_rotation / Num_Rotaion
            Projrction_thisangle_numpy = Model.forward()
            print(time() - time_1)

            # Projrction_numpy = np.squeeze(Temp_Projrction.cpu().detach().numpy())
            ProjNum = i_ProjAngle * PSFSize[3] + j_ProjAngle
            Projrction_thisangle_numpy_fix = Projrction[:, :, ProjNum]

            loss = criterion(Projrction_thisangle_numpy, Projrction_thisangle_numpy_fix)
        # warped_Phantom_numpy = np.squeeze(warped_Phantom.detach().numpy())


            loss.backward()  # 背向传播
            Loss_list[epoch*Num_Rotaion*map.shape[0] + map.shape[0] * i_rotation + i_Map] = loss.item()
            sys.stdout.write("[Train] [Epoch {}/{}] [Rotation {}/{}] [Angle {}/{}] [loss:{:.8f}] time {:.3f}\n"
                             .format(epoch + 1, stop_epoch, i_rotation, Num_Rotaion, i_Map, map.shape[0], loss.item(), time() - time_1))
            sys.stdout.flush()

            optimizer.step()  # 优化器进行更新
            total_loss = total_loss + loss.item()
            Phantom_threshold = torch.zeros_like(Model.Phantom) + threshold

            Model.Phantom.data = torch.where(Model.Phantom.data < threshold, Phantom_threshold, Model.Phantom.data)

        # io.savemat(SavePath + 'Loss', {'Loss': Loss_list})
        writer.add_scalar('training loss', total_loss / map.shape[0], epoch * i_rotation)
        Recon_Phantom_numpy = np.squeeze(Model.Phantom.cpu().detach().numpy())

        tif = TIFF.open(SavePath + 'Recon_Phantom_' + str(epoch) + '_' + str(i_rotation) + '.tif', mode='w')
        for i in range(0, Recon_Phantom_numpy.shape[2]):
            tif.write_image(np.squeeze(Recon_Phantom_numpy[:, :, i]))

        # io.savemat(SavePath + 'Recon_Phantom_' + str(epoch) + '_' + str(i_rotation),
        #            {'a': 1, 'Recon_Phantom_numpy': Recon_Phantom_numpy})

        # warped_Phantom_fix_numpy = np.squeeze(warped_Phantom_fix.cpu().detach().numpy())
        # io.savemat(SavePath + 'warped_Phantom_fix_' + str(epoch) + '_' + str(i_rotation),
        #            {'a': 1, 'warped_Phantom_fix': warped_Phantom_fix_numpy})

        # RefractionIndex_numpy = np.squeeze(Model.RefractionIndex.cpu().detach().numpy())
        # io.savemat(SavePath + 'RefractionIndex_' + str(epoch) + '_' + str(i_rotation),
        #            {'a': 1, 'RefractionIndex': RefractionIndex_numpy})




        # 控制台输出
        # sys.stdout.write("\r[Train] [Epoch {}/{}] [Batch {}/{}] [loss:{:.8f}] [learning rate:{}]"
        #                  .format(epoch + 1, stop_epoch, index + 1, batch_sum,                                                                                                'lr']))


    # 保存权重，每个epoch进行一次保存

# time_1 = time()
# warped_Phantom = Model.forward(RefractionIndex, Phantom)
# print(time() - time_1)

a = 1
