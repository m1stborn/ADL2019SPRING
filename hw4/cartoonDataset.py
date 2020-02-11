import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

def default_loader(path):
    # return Image.open(path).convert('RGB')
    return Image.open(path)

class cartoonDataset(Dataset):
    def __init__(self, txt="./selected_cartoonset100k/cartoon_attr.txt",
                 transform=None, target_transform=None, loader=default_loader,is_train=0):
        imgs = []
        if is_train:
            with open(txt) as f:
                lines = f.readlines()[2:]
            for line in lines:
                attr = line.split()
                imgs.append(( attr[0], list(map(int, attr[1:])) ) )

        self.imgs = imgs
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                            # transforms.
                                            ])
        self.target_transform = target_transform
        self.loader = loader
        self.img_dir = "./selected_cartoonset100k/images/"
    def __getitem__(self, index):
        fn, attr = self.imgs[index]
        img = self.loader(os.path.join(self.img_dir,fn))
        if self.transform is not None:
            img = self.transform(img)
        # return img,attr
        return img,torch.FloatTensor(attr)
    def __len__(self):
        return len(self.imgs)

loader = transforms.Compose([transforms.ToTensor()])

if __name__ == '__main__':
    # img = default_loader("./selected_cartoonset100k/images/cs203627667430349.png")
    # img = loader(img)
    # print(img.size())
    data = cartoonDataset()
    img,attr = data[0]
    print(img.size())
    print(attr)
    print(len(attr))
    print(len(data))
    print(img.sum())
