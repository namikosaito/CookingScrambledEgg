import os
import cv2
import glob
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .argments.resize  import resize_image
from .argments.crop    import random_crop_image
from .argments.distort import random_distort


class ImageFolder(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, img_dir, size, dsize, distort, test, gpu=None):
        self.size = size
        self.dsize = dsize
        self.distort = distort
        self.test = test
        self.gpu = gpu
        self.img_paths = self._get_img_pathes(img_dir)

    def transform(self, img):
        img = resize_image(img, self.size+self.dsize)
        img = random_crop_image(img, self.size, test=self.test)
        if self.distort:
            img_distort = np.copy(img)
            if random.randrange(2):
                img_distort = random_distort(img_distort)
            img_distort = img_distort.astype(np.float32).transpose(2,0,1)
            img_distort /= 255.0
        else:
            img_distort = np.copy(img)
            img_distort = img_distort.astype(np.float32).transpose(2,0,1)
            img_distort /= 255.0
        img = img.astype(np.float32).transpose(2,0,1)
        img /= 255.0

        return img_distort, img
        

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = cv2.imread(path)
        x_im, y_im = self.transform(img)

        return torch.Tensor(x_im), torch.Tensor(y_im)

    def _get_img_pathes(self, img_dir):
        img_pathes = []
        for i,path in enumerate(img_dir):
            root = path.replace(path.split("/")[-1],"")
            with open(path, "r") as fr:
                lines = fr.readlines()
                for ts,line in enumerate(lines):
                    path = line.replace("\n","").split(" ")[0]
                    img_pathes.append(os.path.join(root, path))
                    
        return img_pathes
    

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    IMAGELIST_NAME = "imglist_entire2.dat"
    DATASET_PATH = "/home/mayu/kubo/scrambledegg_ws/dataset_0530"
    ### dataset
    train_path = []
    test_path = []
    for dir in glob.glob(os.path.join(DATASET_PATH,"*")):
        if "train" in dir:
            train_path.append(os.path.join(dir,IMAGELIST_NAME))
        elif "test" in dir:
            test_path.append(os.path.join(dir,IMAGELIST_NAME))


    #dataset = ImageFolder(train_path, size=128, dsize=10, distort=True, test=False)
    #dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    dataset = ImageFolder(train_path, size=128, dsize=10, distort=True, test=True)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)    

    for x_batch, y_batch in dataloader:
        print(x_batch.shape, y_batch.shape)
        
