import torch
import torch.nn.functional as F
import numpy as np

class LossFunction():
    def __init__(self, env):
        self.mu = env.mu
        self.c = env.c

    def eval_l1_loss(self, nI, I, in_log_space=True):
        if in_log_space:
            I = torch.log(self.mu * I + 1.0) / np.log(self.mu + 1.0) * self.c
        else:
            nI = (torch.exp(nI * np.log(self.mu + 1.0) / self.c) - 1.0) / self.mu
        return torch.mean(torch.abs(nI - I))

    def eval_grad_loss(self, nI, dx, dy, in_log_space=True):
        if in_log_space:
            dx = torch.sign(dx) * torch.log(self.mu * torch.abs(dx) + 1.0) / np.log(self.mu + 1.0) * self.c
            dy = torch.sign(dy) * torch.log(self.mu * torch.abs(dy) + 1.0) / np.log(self.mu + 1.0) * self.c
        nI = (torch.exp(nI * np.log(self.mu + 1.0) / self.c) - 1.0) / self.mu
        nIdx = nI[:,:,:,1:] - nI[:,:,:,:-1]
        nIdy = nI[:,:,1:,:] - nI[:,:,:-1,:]
        if in_log_space:
            nIdx = torch.sign(nIdx) * torch.log(self.mu * torch.abs(nIdx) + 1.0) / np.log(self.mu + 1.0) * self.c
            nIdy = torch.sign(nIdy) * torch.log(self.mu * torch.abs(nIdy) + 1.0) / np.log(self.mu + 1.0) * self.c
        return torch.mean(torch.abs(nIdx - dx[:,:,:,:-1])) + torch.mean(torch.abs(nIdy - dy[:,:,:-1,:]))

    def eval_feature_loss(self, nI, grad, feature, I, in_log_space=True):
        # calc the estimated grad
        Fdx = feature[:,:,:,1:] - feature[:,:,:,:-1]
        Fdy = feature[:,:,1:,:] - feature[:,:,:-1,:]
        gradx = torch.sum(grad[:,:,:,:-1] * Fdx, 1, keepdim=True)
        grady = torch.sum(grad[:,:,:-1,:] * Fdy, 1, keepdim=True)
        neg_gradx = torch.sum(grad[:,:,:,1:] * -Fdx, 1, keepdim=True)
        neg_grady = torch.sum(grad[:,:,1:,:] * -Fdy, 1, keepdim=True)

        # calc weights
        weight_x = (I[:,:,:,1:] - I[:,:,:,:-1]) ** 2
        weight_y = (I[:,:,1:,:] - I[:,:,:-1,:]) ** 2
        weight_x = torch.exp(-weight_x * 9)
        weight_y = torch.exp(-weight_y * 9)

        # eval feature loss
        if not in_log_space:
            nI = (torch.exp(nI * np.log(self.mu + 1.0) / self.c) - 1.0) / self.mu
        nIdx = nI[:,:,:,1:] - nI[:,:,:,:-1]
        nIdy = nI[:,:,1:,:] - nI[:,:,:-1,:]
        feature_loss = torch.mean(torch.abs(nIdx - gradx) * weight_x) + torch.mean(torch.abs(nIdx + neg_gradx) * weight_x)
        feature_loss += torch.mean(torch.abs(nIdy - grady) * weight_y) + torch.mean(torch.abs(nIdy + neg_grady) * weight_y)
        return feature_loss
