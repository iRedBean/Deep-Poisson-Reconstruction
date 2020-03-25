import os
import numpy as np
import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, env, training=True):
        self.__training = training
        self.__train_list = []
        self.__test_list = []

        # load training & testing images list
        scenes = os.listdir(env.data_dir)
        for scene in scenes:
            files = os.listdir(os.path.join(env.data_dir, scene))
            temp_list = []
            for fn in files:
                if not '-' in fn:
                    temp_list.append(os.path.join(env.data_dir, scene, fn[:-4]))
            temp_list.sort()
            train_count = int(len(temp_list) * env.training_percent)
            if training:
                self.__train_list.extend(temp_list[:train_count])
            else:
                self.__test_list.extend(temp_list[train_count:])

        if training:
            print('Count of training images:', len(self.__train_list))
        else:
            print('Count of testing images:', len(self.__test_list))

    def __len__(self):
        return len(self.__train_list) * 3 if self.__training else len(self.__test_list) * 3

    def __getitem__(self, index):
        fn = self.__train_list[index // 3] if self.__training else self.__test_list[index // 3]
        # load R/G/B channel
        ch = index % 3

        # load throughput
        I = np.load(fn + '.npy')[:,:,ch:ch+1]
        I = np.transpose(I, (2, 0, 1))
        I = torch.from_numpy(I)

        # load dx & dy
        dx = np.load(fn + '-dx.npy')[:,:,ch:ch+1]
        dy = np.load(fn + '-dy.npy')[:,:,ch:ch+1]
        dx = np.transpose(dx, (2, 0, 1))
        dy = np.transpose(dy, (2, 0, 1))
        dx = torch.from_numpy(dx)
        dy = torch.from_numpy(dy)

        # load feature
        feature = np.load(fn + '-feature.npy')
        feature = np.transpose(feature, (2, 0, 1))
        # normalize depth
        feature[feature > 1e5] = 1e5
        feature[6,:,:] -= feature[6,:,:].min()
        feature[6,:,:] /= feature[6,:,:].max()
        # normalize normal
        feature[3:6,:,:] = feature[3:6,:,:] * 0.5 + 0.5
        feature = torch.from_numpy(feature)

        return I, dx, dy, feature
