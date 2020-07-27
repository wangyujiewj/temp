import sys
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
import ModelProjectionFourier3_7 as Modle_PJ
from libtiff import TIFF

class parameter:
    def __init__(self):
        # Phantom
        self.PhantomSize = []

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

def main():
    ## Parameter
    param = parameter()
    ## GPU or CPU

    param.FlagGrad_RefractionIndex = True
    ## Initialize_PhantomSize
    PhantomSize = [525, 525, 525]
    # Phantom_numpy = Initialize_Phantom(PhantomSize)
    # data = io.loadmat('Phantom_real_525.mat')
    Phantom_numpy = np.zeros(PhantomSize) + 10000
    Phantom = Phantom_numpy.astype('float32')
    Phantom = torch.from_numpy(Phantom)
    ## Initialize_PSF
    mat = h5py.File('PSF/psf_sim_RoundAp_301_301_15_15_525.mat', 'r')
    PSF_numpy = mat['psf']
    PSF_numpy = np.transpose(mat['psf'])
    PSF_numpy = PSF_numpy.astype('float32')
    PSF = torch.from_numpy(PSF_numpy)
    PSFSize = PSF_numpy.shape
    # Projection map
    data = io.loadmat('WJM_Map/wjm_chyf15_map.mat')
    # (109, 2)
    map = data['map']
    ## Rotation
    Num_Rotaion = 24
    center = [0.5, 0.5]
    EdgeRate = 1.42
    temp = np.floor(PhantomSize[0] / EdgeRate / 2)
    ErrorRange = [(PhantomSize[0] + 1) / 2 - temp, (PhantomSize[0] + 1) / 2 + temp]
    ErrorRange = torch.arange(ErrorRange[0], ErrorRange[1] + 1).long()
    ## uploading
    param.center = center
    param.PhantomSize = PhantomSize
    param.PSFSize = PSFSize
    ## Undate Flag
    param.FlagGrad_Phantom = True
    PSF = PSF.cuda()
    param = param.cuda()
    Phantom = Phantom.cuda()
    ## Training Parameter
    Model = Modle_PJ.Model_Projection(Phantom, param, PSF).cuda()
    if torch.cuda.device_count() > 1:
        Model = torch.nn.DataParallel(Model)
    lr = 0.001
    optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
    threshold = 1e-4
    ## Set Path
    SavePath = 'Train_Recon_Rot_' + str(Num_Rotaion) + '_711_NoZero/'
    WignerPath = 'Wigner_Sim/Projection_Fourier6_5_Phantom507_Rot' + str(Num_Rotaion) + '/'
    if not os.path.exists(SavePath): os.mkdir(SavePath)
    ## Training
    stop_epoch = 50
    Loss_list = np.zeros(Num_Rotaion * map.shape[0])

    for i_rotation in range(0, Num_Rotaion):
        data = io.loadmat(WignerPath + 'Projrction_' + str(i_rotation) + '_' + str(Num_Rotaion))
        Projrction = data['Projrction']
        Projrction = Projrction.astype('float32')
        Projrction = torch.from_numpy(Projrction)
        Projrction = Projrction.cuda()
        Model.train()  # 训练模式
        Model.param.theta = math.pi * 2 * i_rotation / Num_Rotaion
        ## Rotation
        # for i in range(0, RotAngle):
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

            loss = F.mse_loss(Projrction_thisangle_numpy, Projrction_thisangle_numpy_fix)
            # warped_Phantom_numpy = np.squeeze(warped_Phantom.detach().numpy())
            loss.backward()  # 背向传播
            Loss_list[Num_Rotaion * map.shape[0] + map.shape[0] * i_rotation + i_Map] = loss.item()
            sys.stdout.write("[Train] [Epoch 1 [Rotation {}/{}] [Angle {}/{}] [loss:{:.8f}] time {:.3f}\n"
                             .format(i_rotation, Num_Rotaion, i_Map, map.shape[0],
                                     loss.item(), time() - time_1))
            sys.stdout.flush()

            optimizer.step()  # 优化器进行更新

            Phantom_threshold = torch.zeros_like(Model.Phantom) + threshold

            Model.Phantom.data = torch.where(Model.Phantom.data < threshold, Phantom_threshold, Model.Phantom.data)

            io.savemat(SavePath + 'Loss', {'Loss': Loss_list})

            Recon_Phantom_numpy = np.squeeze(Model.Phantom.cpu().detach().numpy())

            tif = TIFF.open(SavePath + 'Recon_Phantom_' + '_' + str(i_rotation) + '.tif', mode='w')
            for i in range(0, Recon_Phantom_numpy.shape[2]):
                tif.write_image(np.squeeze(Recon_Phantom_numpy[:, :, i]))

            io.savemat(SavePath + 'Recon_Phantom_' + '_' + str(i_rotation),
                       {'a': 1, 'Recon_Phantom_numpy': Recon_Phantom_numpy})
if __name__ == '__main__':
    main()
