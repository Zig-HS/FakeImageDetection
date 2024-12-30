
import random
import numpy as np
import logging
import torch

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import average_precision_score, accuracy_score

from data.datasets import ResTransformerDataset
from models.trainer import Trainer
from utils import utils_logger
from utils.utils_tools import mkdir, load_options, dict2str
from utils.utils_earlystop import EarlyStopping

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate(model, opt):
    valset_root = f"dataset/val/{opt['datasets']['name']}"
    val_dataset = ResTransformerDataset(
        opt=opt,
        mode='val',
        root=valset_root,
        transform=None,
    )
    val_dataloder = DataLoader(
        dataset=val_dataset,
        batch_size=opt['datasets']['batch_size'],
        shuffle=opt['datasets']['shuffle'],
        num_workers=opt['datasets']['num_workers'],
        drop_last=True,
        pin_memory=True
    )
    with torch.no_grad():
        y_true, y_pred = [], []
        for i, data in enumerate(val_dataloder):
            input = []
            for patch_L in data['L']:
                input.append(patch_L.cuda())
            output = model(input)
            y_pred.extend(output['label'].sigmoid().flatten().tolist())
            y_true.extend(data['label'].flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

def main(json_path):

    # initialize opt
    opt=load_options(json_path, is_train=True)
    mkdir(opt['path'])
    mkdir(opt['models_path'])

    # Load Logger
    logger_name = opt['name']
    utils_logger.logger_info(logger_name, opt['logger_path'])
    logger = logging.getLogger(logger_name)
    logger.info(dict2str(opt))

    # Set Seed
    seed_everything(opt['seed'])

    # Load Dataset
    trainset_root = f"dataset/train/{opt['datasets']['name']}"
    train_dataset = ResTransformerDataset(
        opt=opt,
        mode='train',
        root=trainset_root,
        transform=None,
    )
    train_dataset_size=len(train_dataset)
    logger.info(f"Training Images = {train_dataset_size}")

    # Load Model
    model = Trainer(opt)

    # Train
    indices = torch.randperm(train_dataset_size).tolist()
    subset_indices = indices[:opt['datasets']['num_samples']]
    sampler = SubsetRandomSampler(subset_indices)
    train_dataloder = DataLoader(
        dataset=train_dataset,
        batch_size=opt['datasets']['batch_size'],
        num_workers=opt['datasets']['num_workers'],
        drop_last=True,
        pin_memory=True,
        sampler=sampler,
    )

    logger.info(f"Training Batches = {len(train_dataloder)}")

    # Start Training
    early_stopping = EarlyStopping(opt, patience=opt['train']['early_stop_epoch'], delta=-0.001)
    for epoch in range(opt['train']['niter']):


        logger.info(f"Start @ epoch {epoch}")
        for i, data in enumerate(train_dataloder):
            model.D_steps += 1 * opt['datasets']['batch_size'] # for update learning rate
            model.C_steps += 1
            model.D_adjust_learning_rate()
            model.set_input(data)
            model.optimize_parameters()
            print(f"Steps: {model.C_steps} D_loss:{model.D_loss} C_loss: {model.C_loss}",end='\r')
            if model.C_steps % opt['train']['show_loss_freq'] == 0:
                logger.info(f"D_loss: {model.D_loss} C_loss: {model.C_loss} at C_steps-{model.C_steps:6};")
            if model.C_steps % opt['train']['save_latest_freq'] == 0:
                model.save_network('latest')
        if epoch % opt['train']['save_epoch_freq'] == 0:
            model.save_network(epoch)
            model.save_network('latest')

        # Validate
        model.eval()
        acc, ap = validate(model.model, opt)[:2]
        logger.info(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.C_adjust_learning_rate()
            if cont_train:
                logger.info("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(opt, patience=opt['train']['early_stop_epoch'], delta=-0.001)
            else:
                logger.info("Early stopping.")
                break
        model.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-j','--json', type=str, default='options/default.json')
    opt = parser.parse_args()
    main(opt.json)

