
import os
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from models.select_model import select_network
from data.datasets import ResTransformerDataset
from utils.utils_tools import load_options

def main(json_path,test_root):

    # Load Options
    opt=load_options(json_path, gpu_list='0')
    opt['is_train'] = False

    # Load Model
    device = torch.device(f"cuda")
    model = select_network(opt).to(device)
    load_path = os.path.join(opt['models_path'], 'epoch_best.pth')
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict, strict=True)

    # Load Data
    test_sets = os.listdir(test_root)
    test_sets = sorted(test_sets)
    print(test_sets)
    for test_set in test_sets:
        print(f"Testing on {test_set}")
        test_path = os.path.join(test_root, test_set)
        val_batch_size = opt['datasets']['batch_size']
        val_dataset = ResTransformerDataset(
            opt=opt,
            mode='val',
            root=test_path,
            transform=None,
        )
        val_dataloder = DataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=opt['datasets']['shuffle'],
            num_workers=opt['datasets']['num_workers'],
            drop_last=True,
            pin_memory=True
        )
        with torch.no_grad():
            y_true, y_pred = [], []
            for data in tqdm(val_dataloder):
                input = []
                for patch_L in data['L']:
                    input.append(patch_L.cuda())
                output = model(input)
                y_pred.extend(output['label'].sigmoid().flatten().tolist())
                y_true.extend(data['label'].flatten().tolist())
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        with open(f"results/{opt['name']}.txt", 'a') as f:
            f.write(f"AUC:{auc*100:2.2f}\tAP:{ap*100:2.2f}\tTest-set:{test_set}\n")
            print(f"AUC:{auc*100:2.2f}\tAP:{ap*100:2.2f}\tTest-set:{test_set}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-j','--json', type=str, default='options/default.json')
    parser.add_argument('-r','--root', type=str, default='dataset/test')
    opt = parser.parse_args()
    main(opt.json, opt.root)

