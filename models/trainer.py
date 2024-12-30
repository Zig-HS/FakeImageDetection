import os
import logging
import torch
import torch.nn as nn

from models.select_model import select_network

class Trainer(nn.Module):
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.opt        = opt
        self.is_train   = opt['is_train']
        self.D_steps    = 0
        self.C_steps    = 0
        self.schedulers = []
        self.save_dir   = opt['models_path']
        self.device     = torch.device(f"cuda")
        self.logger     = logging.getLogger(opt['name'])
        self.model      = select_network(opt).to(self.device)

        if self.is_train:
            self.init_optimizer()
        # if not self.is_train or opt['continue_train']:
        #     self.load_network(opt.epoch) # TODO

    def save_network(self, epoch):
        save_filename = f'epoch_{epoch}.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = self.model.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, strict=True):
        self.model.load_state_dict(torch.load(load_path), strict=strict)

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def init_optimizer(self):
        # Denoiser
        self.D_lossfn = nn.L1Loss().to(self.device)
        self.D_lossfn_weight = self.opt['train']['D_lossfn_weight']
        self.D_optimizer = torch.optim.Adam(
            [
                {'params': self.model.denoiser.parameters()},
                {'params': self.model.return_image.parameters()},
            ],
            lr=self.opt['train']['D_learing_rate'],
            weight_decay=0,
        )
        self.scheduler=torch.optim.lr_scheduler.MultiStepLR(
            self.D_optimizer,
            self.opt['train']['D_scheduler'],
            self.opt['train']['D_scheduler_gamma'],
        )
        # Classifier
        if self.opt['trainer'] == "1loss":
            self.C_loss_fn = nn.BCEWithLogitsLoss()
            self.C_optimizer = torch.optim.Adam(
                [
                    {'params': self.model.parameters()},
                ],
                lr=self.opt['train']['C_learning_rate'],
                betas=(self.opt['train']['beta1'], 0.999),
            )
        else:
            self.C_loss_fn = nn.BCEWithLogitsLoss()
            self.C_optimizer = torch.optim.Adam(
                [
                    {'params': self.model.patches_to_embedding.parameters()},
                    {'params': self.model.ReT.parameters()},
                ],
                lr=self.opt['train']['C_learning_rate'],
                betas=(self.opt['train']['beta1'], 0.999),
            )

    def D_adjust_learning_rate(self):
        for i in range(self.opt['datasets']['batch_size']):
            self.scheduler.step()

    def C_adjust_learning_rate(self, min_lr=1e-6):
        # Classifier
        for param_group in self.C_optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input, need_H=True):
        self.L = []
        for patch_L in input['L']:
            self.L.append(patch_L.to(self.device))
        self.H = []
        for patch_H in input['H']:
            self.H.append(patch_H.to(self.device))
        self.label = input['label'].to(self.device).float()

    def forward(self):
        output = self.model.forward(self.L)
        self.E = output['E']
        self.output = output['label']

    def optimize_parameters(self):
        if self.opt['trainer'] == "1loss":
            self.L = self.H
            self.forward()
            self.C_optimizer.zero_grad()
            self.C_loss = self.C_loss_fn(self.output.squeeze(1), self.label)
            self.C_loss.backward()
            self.C_optimizer.step()
            self.D_loss = 'None'
        elif self.opt['trainer'] == "2step":
            self.forward()
            self.D_optimizer.zero_grad()
            D_losses = []
            for patch_E,patch_H in zip(self.E,self.H):
                D_loss = self.D_lossfn_weight * self.D_lossfn(patch_E, patch_H)
                D_losses.append(D_loss)
            self.D_loss = sum(D_losses)/len(D_losses)
            self.D_loss.backward()
            self.D_optimizer.step()
            self.L = self.H
            self.forward()
            self.C_optimizer.zero_grad()
            self.C_loss = self.C_loss_fn(self.output.squeeze(1), self.label)
            self.C_loss.backward()
            self.C_optimizer.step()
        else:
            self.forward()
            self.D_optimizer.zero_grad()
            self.C_optimizer.zero_grad()
            D_losses = []
            for patch_E,patch_H in zip(self.E,self.H):
                D_loss = self.D_lossfn_weight * self.D_lossfn(patch_E, patch_H)
                D_losses.append(D_loss)
            self.D_loss = sum(D_losses)/len(D_losses)
            self.D_loss.backward(retain_graph=True)
            self.C_loss = self.C_loss_fn(self.output.squeeze(1), self.label)
            self.C_loss.backward()
            self.D_optimizer.step()
            self.C_optimizer.step()







