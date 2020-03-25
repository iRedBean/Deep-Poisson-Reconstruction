import torch
from networks import GNet
from loss_functions import LossFunction

class GBranch():
    def __init__(self, env):
        self.model = GNet()
        self.model = torch.nn.DataParallel(self.model)
        self.loss_func = env.loss_func
        if env.cuda:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=env.lr, betas=(env.beta, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, env.decay_epoch, env.decay_rate)

    def set_training(self, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()

    def scheduler_step(self):
        self.scheduler.step()

    def predict(self, x):
        return self.model(x)

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, nI, grad, feature, I):
        self.optimizer.zero_grad()
        loss = self.loss_func.eval_feature_loss(nI.detach(), grad, feature, I)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
