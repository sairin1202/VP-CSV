import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob
import numpy as np
import random
from vocab import Vocabulary
import pickle
from collections import Counter
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
import torch

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


def load_image(file):
    return Image.open(file)

print('vocab:', len(vocab))

class train_dataset(Dataset):
    def __init__(self):
        with open('data/train_data.pkl','rb') as f:
            self.train_data = pickle.load(f)
          
        self.input_transform = Compose([Resize((224, 224)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        

    def __getitem__(self, index):

        (im, tok) = self.train_data[index]
        im = im.split('/')[-1]
        im = f"/scratch/acb11361bd/StoryGan/PlanAndDraw-Pororo/data/train/{im}"
        with open(im, "rb") as f:
            image = load_image(f).convert('RGB')

        # random horizontal flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.input_transform is not None:
            image = self.input_transform(image)
        return image, torch.LongTensor([vocab(tok)])

    def __len__(self):
        return len(self.train_data)



class test_dataset(Dataset):
    def __init__(self):
        with open('data/test_data.pkl','rb') as f:
            self.test_data = pickle.load(f)
          
        self.input_transform = Compose([Resize((224, 224)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
        

    def __getitem__(self, index):

        (im, tok) = self.test_data[index]
        im = im.split('/')[-1]
        im = f"/scratch/acb11361bd/StoryGan/PlanAndDraw-Pororo/data/train/{im}"
        with open(im, "rb") as f:
            image = load_image(f).convert('RGB')

        # random horizontal flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.input_transform is not None:
            image = self.input_transform(image)

        return image, torch.LongTensor([vocab(tok)])

    def __len__(self):
        return len(self.test_data)
