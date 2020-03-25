import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kernel_1D(r, sigma):
    kernel = np.zeros(2*r+1, dtype=np.float32)
    for x in range(-r, r+1):
        kernel[x+r] = np.exp(-x**2/(2*sigma**2))
    kernel /= np.sum(kernel)
    return kernel

class PostProcessFilter():
    def __init__(self, radius, sigma, cuda=True):
        self.x_kernel = torch.from_numpy(np.tile(gaussian_kernel_1D(radius, sigma), 9).reshape(3, 3, 1, 2*radius+1))
        self.y_kernel = torch.from_numpy(np.tile(gaussian_kernel_1D(radius, sigma), 9).reshape(3, 3, 2*radius+1, 1))
        self.radius = radius
        if cuda:
            self.x_kernel = self.x_kernel.cuda()
            self.y_kernel = self.y_kernel.cuda()

    def process(self, I, nI):
        I = I.unsqueeze(0)
        nI = nI.unsqueeze(0)
        _I = F.conv2d(I, self.x_kernel, padding=(0, self.radius))
        _I = F.conv2d(_I, self.y_kernel, padding=(self.radius, 0)) + 1e-9
        _nI = F.conv2d(nI, self.x_kernel, padding=(0, self.radius))
        _nI = F.conv2d(_nI, self.y_kernel, padding=(self.radius, 0)) + 1e-9
        nI *= _I / _nI
        return nI.squeeze(0)
