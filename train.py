import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from process_data import Dataset
from networks import GradNet
from g_branch import GBranch
from loss_functions import LossFunction

def eval_loss(env, nI, grad, I, dx, dy, feature):
    L1_loss = env.loss_func.eval_l1_loss(nI, I)
    Grad_loss = env.loss_func.eval_grad_loss(nI, dx, dy)
    Feature_loss = env.loss_func.eval_feature_loss(nI, grad.detach(), feature, I)
    return env.lambda_L1 * L1_loss, Grad_loss, env.lambda_F * Feature_loss

def train_g_model(env):
    env.g_model.set_training(True)
    train_loss = {'Feature_loss': 0.0}
    for I, dx, dy, feature in env.train_loader:
        if env.cuda:
            I, dx, dy, feature = I.cuda(), dx.cuda(), dy.cuda(), feature.cuda()
        I, dx, dy, feature = Variable(I), Variable(dx), Variable(dy), Variable(feature)
        # image & feature
        I_input = torch.log(env.mu * I + 1.0) / np.log(env.mu + 1.0) * env.c
        I_input = torch.cat([I_input, feature], 1)
        # dx & dy
        grad_input = torch.cat([dx, dy], 1)
        grad_input = torch.sign(grad_input) * torch.log(env.mu * torch.abs(grad_input) + 1.0) / np.log(env.mu + 1.0) * env.c
        # estimated grad
        grad = env.g_model.predict(torch.cat([I_input, grad_input], 1))
        # eval feature loss with reconstructed nI
        # nI = env.model(I_input, grad_input)
        # train_loss['Feature_loss'] += env.g_model.update(nI, grad, feature, I)
        # eval feature loss with original I
        train_loss['Feature_loss'] += env.g_model.update(torch.log(env.mu * I + 1.0) / np.log(env.mu + 1.0) * env.c, grad, feature, I)
    # print info
    for key in train_loss.keys():
        train_loss[key] /= len(env.train_loader)
    env.train_loss_recorder['G-branch'].append(train_loss['Feature_loss'])
    print('Training --- Feature_loss: {Feature_loss:.6f}'.format(**train_loss))

def train(env):
    env.model.train()
    env.g_model.set_training(True)
    train_loss = {'L1_loss': 0.0, 'Grad_loss': 0.0, 'Feature_loss': 0.0}
    for I, dx, dy, feature in env.train_loader:
        if env.cuda:
            I, dx, dy, feature = I.cuda(), dx.cuda(), dy.cuda(), feature.cuda()
        I, dx, dy, feature = Variable(I), Variable(dx), Variable(dy), Variable(feature)
        # image & feature
        I_input = torch.log(env.mu * I + 1.0) / np.log(env.mu + 1.0) * env.c
        I_input = torch.cat([I_input, feature], 1)
        # dx & dy
        grad_input = torch.cat([dx, dy], 1)
        grad_input = torch.sign(grad_input) * torch.log(env.mu * torch.abs(grad_input) + 1.0) / np.log(env.mu + 1.0) * env.c
        # reconstructed nI
        nI = env.model(I_input, grad_input)
        # estimated grad
        grad = env.g_model.predict(torch.cat([I_input, grad_input], 1))
        # eval loss
        env.optimizer.zero_grad()
        L1_loss, Grad_loss, Feature_loss = eval_loss(env, nI, grad, I, dx, dy, feature)
        loss = L1_loss + Grad_loss + Feature_loss
        train_loss['L1_loss'] += L1_loss.item()
        train_loss['Grad_loss'] += Grad_loss.item()
        train_loss['Feature_loss'] += Feature_loss.item()
        loss.backward()
        env.optimizer.step()
    # print info
    for key in train_loss.keys():
        train_loss[key] /= len(env.train_loader)
    env.train_loss_recorder['L1 loss'].append(train_loss['L1_loss'])
    env.train_loss_recorder['Grad loss'].append(train_loss['Grad_loss'])
    env.train_loss_recorder['Feature loss'].append(train_loss['Feature_loss'])
    print('Training --- L1_loss: {L1_loss:.6f}, Grad_loss: {Grad_loss:.6f}, Feature_loss: {Feature_loss:.6f}'.format(**train_loss))

def test(env):
    env.model.eval()
    env.g_model.set_training(False)
    test_loss = {'L1_loss': 0.0, 'Grad_loss': 0.0, 'Feature_loss': 0.0}
    for I, dx, dy, feature in env.test_loader:
        if env.cuda:
            I, dx, dy, feature = I.cuda(), dx.cuda(), dy.cuda(), feature.cuda()
        I, dx, dy, feature = Variable(I), Variable(dx), Variable(dy), Variable(feature)
        # image & feature
        I_input = torch.log(env.mu * I + 1.0) / np.log(env.mu + 1.0) * env.c
        I_input = torch.cat([I_input, feature], 1)
        # dx & dy
        grad_input = torch.cat([dx, dy], 1)
        grad_input = torch.sign(grad_input) * torch.log(env.mu * torch.abs(grad_input) + 1.0) / np.log(env.mu + 1.0) * env.c
        # reconstructed nI
        nI = env.model(I_input, grad_input)
        # estimated grad
        grad = env.g_model.predict(torch.cat([I_input, grad_input], 1))
        # eval loss
        L1_loss, Grad_loss, Feature_loss = eval_loss(env, nI, grad, I, dx, dy, feature)
        test_loss['L1_loss'] += L1_loss.item()
        test_loss['Grad_loss'] += Grad_loss.item()
        test_loss['Feature_loss'] += Feature_loss.item()
    # print info
    for key in test_loss.keys():
        test_loss[key] /= len(env.test_loader)
    env.test_loss_recorder['L1 loss'].append(test_loss['L1_loss'])
    env.test_loss_recorder['Grad loss'].append(test_loss['Grad_loss'])
    env.test_loss_recorder['Feature loss'].append(test_loss['Feature_loss'])
    print('Testing --- L1_loss: {L1_loss:.6f}, Grad_loss: {Grad_loss:.6f}, Feature_loss: {Feature_loss:.6f}'.format(**test_loss))

def save_loss_figure(env):
    titles = ['L1 loss', 'Grad loss', 'Feature loss', 'G-branch']
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    for idx, title in enumerate(titles):
        subfig = fig.add_subplot(2, 2, idx + 1)
        subfig.set_title(title)
        if title in env.train_loss_recorder:
            subfig.plot(env.train_loss_recorder[title], label='Train')
        if title in env.test_loss_recorder:
            subfig.plot(env.test_loss_recorder[title], label='Test')
        subfig.set_xlabel('Epoch')
        subfig.legend()
    fig.savefig('./loss.png', format='png')
    plt.close(fig)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Deep Poisson Reconstruction')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=71)
    parser.add_argument('--process_count', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--decay_rate', type=float, default=0.95)
    parser.add_argument('--decay_epoch', type=float, default=2)
    parser.add_argument('--lambda_L1', type=float, default=1)
    parser.add_argument('--lambda_F', type=float, default=2)
    parser.add_argument('--mu', type=float, default=16.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--training_percent', type=float, default=0.9)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=131)
    parser.add_argument('--save_interval', type=int, default=5)
    env = parser.parse_args()

    # init
    env.cuda = not env.no_cuda and torch.cuda.is_available()
    torch.manual_seed(env.seed)
    env.model = GradNet()
    env.loss_func = LossFunction(env)
    env.train_loss_recorder = {'L1 loss': [], 'Grad loss': [], 'Feature loss': [], 'G-branch': []}
    env.test_loss_recorder = {'L1 loss': [], 'Grad loss': [], 'Feature loss': []}
    env.g_model = GBranch(env)
    env.g_model.freeze()
    if env.cuda:
        torch.cuda.manual_seed(env.seed)
        env.model = torch.nn.DataParallel(env.model)
        env.model.cuda()
    env.train_loader = torch.utils.data.DataLoader(Dataset(env, training=True), batch_size=env.batch_size, shuffle=True, num_workers=env.process_count)
    env.test_loader = torch.utils.data.DataLoader(Dataset(env, training=False), batch_size=env.batch_size, shuffle=False, num_workers=env.process_count)
    env.optimizer = torch.optim.Adam(env.model.parameters(), lr=env.lr, betas=(env.beta, 0.999))
    env.scheduler = torch.optim.lr_scheduler.StepLR(env.optimizer, env.decay_epoch, env.decay_rate)

    # train & test
    lambda_F = env.lambda_F
    env.lambda_F = 0.0
    for epoch in range(env.epochs):
        env.cur_epoch = epoch
        print('Epoch %d' % (epoch + 1))
        train(env)
        test(env)
        print('Train G-branch.')
        env.g_model.unfreeze()
        train_g_model(env)
        env.g_model.freeze()
        print('--------------------')
        env.scheduler.step()
        env.g_model.scheduler_step()
        if epoch % env.save_interval == 0:
            torch.save(env.model.state_dict(), os.path.join(env.save_dir, 'model(epoch_%d).pkl' % (epoch + 1)))
            env.g_model.save(os.path.join(env.save_dir, 'g_model(epoch_%d).pkl' % (epoch + 1)))
        # adjust the weight of feature loss
        if epoch >= 5:
            env.lambda_F = min(max(env.lambda_F, 0.1) * 1.1, lambda_F)
        save_loss_figure(env)
