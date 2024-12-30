
import cv2
import numpy as np
import torch
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from io import BytesIO
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True

def imread_uint(path, n_channels=3):
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path)  # BGR or G
        try:
            if img.dtype != "uint8":
                img = cv2.convertScaleAbs(img)
        except:
            with open(path, 'rb') as f:
                f=f.read()
            f=f+B'\xff'+B'\xd9'
            img = Image.open(BytesIO(f)).convert("RGB").save(f"temp/temp.png")
            img = cv2.imread(f"temp/temp.png")
            if img.dtype != "uint8":
                img = cv2.convertScaleAbs(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

def zero_pad(image, target_height, target_width):
    h, w, _ = image.shape
    pad_h = max(0, target_height - h)
    pad_w = max(0, target_width - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    return padded_image

def resize(image, loadsize):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w < h:
        new_w = loadsize
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = loadsize
        new_w = int(new_h * aspect_ratio)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def random_crop(img, cropsize):
    h, w, _ = img.shape
    top = np.random.randint(0, h - cropsize + 1)
    left = np.random.randint(0, w - cropsize + 1)
    img = img[top:top+cropsize, left:left+cropsize, :]
    return img

def central_crop(img, cropsize):
    h, w, _ = img.shape
    h_c, w_c = h // 2, w // 2
    top = h_c - cropsize // 2
    left = w_c - cropsize // 2
    img = img[top:top+cropsize, left:left+cropsize, :]
    return img

def jpg_compress(img, compress_val):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def augment_img(img, augument_type=0):
    if augument_type == 0:
        return img
    elif augument_type == 1:
        return np.flipud(np.rot90(img))
    elif augument_type == 2:
        return np.flipud(img)
    elif augument_type == 3:
        return np.rot90(img, k=3)
    elif augument_type == 4:
        return np.flipud(np.rot90(img, k=2))
    elif augument_type == 5:
        return np.rot90(img)
    elif augument_type == 6:
        return np.rot90(img, k=2)
    elif augument_type == 7:
        return np.flipud(np.rot90(img, k=3))

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

def get_patches(img, size, padding):
    h, w, _ = img.shape
    sub_images = []
    for bottom in range(0, h - size + 1, padding):
        for left in range(0, w - size + 1, padding):
            sub_image = img[left:left + size, bottom:bottom + size]
            sub_image = uint2tensor3(sub_image)
            sub_images.append(sub_image)
    return sub_images


class ResTransformerDataset(datasets.ImageFolder):
    def __init__(self, opt, mode, root, transform,):
        super().__init__(root=root, transform=transform,)
        self.mode = mode
        if opt['name']=='RT-nodncnn-dm':
            self.mode = 'val'
        self.imgs = self.samples

        self.resize     = opt['train']['resize']
        self.loadsize   = opt['train']['loadsize']
        self.cropsize   = opt['train']['cropsize']
        self.jpg        = opt['train']['jpg']
        self.augument   = opt['train']['augument']
        self.sigma      = opt['train']['sigma']

        self.patch_size    = opt['network']['patch_size']
        self.padding_size  = opt['network']['padding_size']

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = imread_uint(path)

        if self.mode == 'train':
            if self.resize == True:
                sample = resize(sample, self.loadsize)
            else:
                sample = zero_pad(sample, self.loadsize, self.loadsize)
            sample = random_crop(sample, self.cropsize)
            if self.jpg == True:
                if random.random() < 0.5:
                    sample = jpg_compress(sample,random.randint(30,100))
                if random.random() < 0.5:
                    gaussian_blur(sample,random.random()*3.0)
            if self.augument == True:
                augument_type = np.random.randint(0, 8)
                sample = augment_img(sample, augument_type=augument_type)
            patches_L = []
            patches_H = get_patches(sample, size=self.patch_size, padding=self.padding_size)
            for patch_H in patches_H:
                noise = torch.randn(patch_H.size()).mul_(self.sigma/255.0)
                patch_L = patch_H.clone()
                patch_L.add_(noise)
                patches_L.append(patch_L)
        elif self.mode == 'val':
            if self.resize == True:
                sample = resize(sample, self.loadsize)
            else:
                sample = zero_pad(sample, self.loadsize, self.loadsize)
            sample = central_crop(sample, self.cropsize)
            patches_H = get_patches(sample, size=self.patch_size, padding=self.padding_size)
            patches_L = patches_H

        return {'L': patches_L, 'H': patches_H, 'label': label}
