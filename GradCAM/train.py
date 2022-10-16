from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch.optim as optim
from dataset import train_dataset, test_dataset
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
import pickle
from tqdm import tqdm
import torch
import timm

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
# print(model)
model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]
model.fc = nn.Linear(2048, len(vocab))
# model.head = nn.Linear(2048, len(vocab))
model = model.cuda()
model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=4e-5)
loss_func = nn.CrossEntropyLoss()

train_data_loader = DataLoader(train_dataset(), num_workers=64, batch_size=1024, shuffle=True)
test_data_loader = DataLoader(test_dataset(), num_workers=64, batch_size=1024, shuffle=False)

def evaluate(model, epoch):
    model.eval()
    losses = []
    for i, (image, label) in enumerate(tqdm(test_data_loader)):
        image, label = image.cuda(), label.cuda()
        logit = model(image)
        loss = loss_func(logit, label.view(-1))
        losses.append(loss.item())

    print("---------------------------------------------")
    print("testing loss: {}".format(np.mean(losses)))
    model.train()



for epoch in range(100):
    losses = []
    for i, (image, label) in enumerate(tqdm(train_data_loader)):
        image = image.cuda()
        label = label.cuda()
        logit = model(image)
        loss = loss_func(logit, label.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print("---------------------------------------------")
    print("training loss, epoch{} iter {}/{} loss: {}".format(epoch, i, len(train_data_loader), np.mean(losses)))
    print("testing......")
    evaluate(model, epoch)
    if epoch % 5 == 0:
        torch.save(model.module.state_dict(), f'model/epoch{epoch}.pt')
