import os
import json
from collections import OrderedDict


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_options(json_path, is_train=True, gpu_list=None):
    json_str = ''
    with open(json_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt['path']         = os.path.join(opt['root'], opt['name'])
    opt['models_path']  = os.path.join(opt['path'], "models")
    opt['logger_path']  = f"{opt['path']}/{opt['name']}.log"
    opt['is_train'] = is_train

    try:
        opt['trainer'] = opt['trainer']
    except KeyError:
        opt['trainer'] = 'default'

    if gpu_list == None:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    return opt


def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg