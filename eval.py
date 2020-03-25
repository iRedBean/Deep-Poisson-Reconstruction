import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from networks import GradNet
from post_process import PostProcessFilter
import python_pfm

def load_test_image(file_dir, fid):
    if os.path.exists(os.path.join(file_dir, '{}-throughput.npy'.format(fid))):
        I = np.load(os.path.join(file_dir, '{}-throughput.npy'.format(fid)))
    else:
        I = None

    if os.path.exists(os.path.join(file_dir, '{}-dx.npy'.format(fid))):
        dx = np.load(os.path.join(file_dir, '{}-dx.npy'.format(fid)))
    else:
        dx = None

    if os.path.exists(os.path.join(file_dir, '{}-dy.npy'.format(fid))):
        dy = np.load(os.path.join(file_dir, '{}-dy.npy'.format(fid)))
    else:
        dy = None

    if os.path.exists(os.path.join(file_dir, '{}-feature.npy'.format(fid))):
        feature = np.load(os.path.join(file_dir, '{}-feature.npy'.format(fid)))
    else:
        feature = None

    return I, dx, dy, feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Poisson Reconstruction')
    parser.add_argument('--data_dir', type=str, default='./test_data')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--epoch', type=int, default=11)
    parser.add_argument('--mu', type=float, default=16.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--spp_l', type=int, default=5)
    parser.add_argument('--spp_r', type=int, default=9)
    parser.add_argument('--pp_radius', type=int, default=45)
    parser.add_argument('--pp_sigma', type=float, default=15.0)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    env = parser.parse_args()

    # load model
    env.cuda = not env.no_cuda and torch.cuda.is_available()
    model = GradNet()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(env.save_dir, 'model(epoch_%d).pkl' % env.epoch)))
    if env.cuda:
        model.cuda()
    model.eval()

    pp_filter = PostProcessFilter(env.pp_radius, env.pp_sigma, cuda=env.cuda)

    # for each scene
    scenes = os.listdir(env.data_dir)
    for scene in scenes:
        print('\nCurrent:', scene)
        file_dir = os.path.join(env.data_dir, scene)
        ref_image = python_pfm.load_pfm(os.path.join(file_dir, '{}-gt.pfm'.format(scene)))

        for i in range(env.spp_l, env.spp_r + 1):
            test_fn = '{}-{}'.format(scene, 2 ** i)
            I, dx, dy, feature = load_test_image(file_dir, test_fn)
            if (I is None) or (dx is None) or (dy is None) or (feature is None):
                continue
            r = I.shape[0]
            c = I.shape[1]

            st = time.time()

            I = np.transpose(I, (2, 0, 1))
            dx = np.transpose(dx, (2, 0, 1))
            dy = np.transpose(dy, (2, 0, 1))
            feature = np.transpose(feature, (2, 0, 1))
            feature[feature > 1e5] = 1e5
            feature[6,:,:] -= feature[6,:,:].min()
            feature[6,:,:] /= feature[6,:,:].max()
            feature[3:6,:,:] = feature[3:6,:,:] * 0.5 + 0.5

            I = torch.from_numpy(I)
            dx = torch.from_numpy(dx)
            dy = torch.from_numpy(dy)
            feature = torch.from_numpy(feature)

            if env.cuda:
                I, dx, dy, feature = I.cuda(), dx.cuda(), dy.cuda(), feature.cuda()

            I = I.unsqueeze(1)
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            feature = feature.unsqueeze(0)
            feature = feature.expand(3, 7, r, c)

            I_input = torch.log(env.mu * I + 1.0) / np.log(env.mu + 1.0) * env.c
            I_input = torch.cat([I_input, feature], 1)
            grad_input = torch.cat([dx, dy], 1)
            grad_input = torch.log(env.mu * torch.abs(grad_input) + 1.0) / np.log(env.mu + 1.0) * env.c * torch.sign(grad_input)
            nI = model(I_input, grad_input)
            nI = (torch.exp(nI * np.log(env.mu + 1.0) / env.c) - 1.0) / env.mu
            nI = nI.squeeze(1)

            nI = pp_filter.process(I.squeeze(1), nI)

            nI = nI.cpu().detach().numpy()
            nI = np.transpose(nI, (1, 2, 0))

            ed = time.time()
            print('Reconstruction cost:', (ed - st) * 1000, 'ms')
            print('relMSE:', np.mean((nI - ref_image)**2 / (ref_image**2 + 0.01)))

            python_pfm.writePFM(os.path.join(file_dir, '{}-output.pfm'.format(test_fn)), nI)
