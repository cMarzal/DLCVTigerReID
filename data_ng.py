import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)


class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        data_dir = args.data_dir
        csv_path = os.path.join(data_dir, mode + '.csv')
        img_dir = os.path.join(data_dir, 'imgs')

        data_F = read_csv(csv_path)
        self.imgs_names = read_csv(csv_path)[1]
        for i in range(len(data_F)):
            data_F[1][i] = os.path.join(img_dir, data_F[1][i])

        self.data = data_F
        img_size = 224
        ''' set up image transform '''
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        self.transform_t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        if self.mode == 'train':
            img = Image.open(self.data[1][idx]).convert('RGB')
            images = self.transform(img)
            cls = self.data[0][idx]

        else:
            img = Image.open(self.data[1][idx]).convert('RGB')
            images = self.transform_t(img)
            cls = self.imgs_names[idx]

        ''' read image '''
        return images, cls

