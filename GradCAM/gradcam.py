from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from vocab import Vocabulary
import pickle
import PIL.Image as Image
import torch
import torch.nn as nn
import cv2
import numpy as np
import timm
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


# def reshape_transform(tensor, height=7, width=7):
#     result = tensor.reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

def get_image(im):

    input_transform = Compose([Resize((224, 224)), ToTensor(
        ), Normalize([.485, .456, .406], [.229, .224, .225])])

    with open(im, "rb") as f:
        image = Image.open(f).convert('RGB')

    image = input_transform(image)
    return image

model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]
model.fc = nn.Linear(2048, len(vocab))
# model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
# model.head = nn.Linear(1024, len(vocab))
model.load_state_dict(torch.load("/scratch/acb11361bd/StoryGan/GradCam-Pororo/model/epoch30.pt"))
print(model)
target_layers = [model.layer4[-1]]
# target_layers = [model.layers[-1].blocks[-1].norm1]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
# cam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=True)


# ----------------------------------------------------------------------------------
# image_path = "/scratch/acb11361bd/StoryGan/METER-Pororo-Single/data/48.png"
# input_tensor = get_image(image_path).unsqueeze(0)
# # print(input_tensor.size())
# rgb_image = cv2.imread(image_path)[:, :, ::-1]
# rgb_image = cv2.resize(rgb_image, (224, 224))
# rgb_image = np.float32(rgb_image) / 255


# for token in ['pororo', 'egg']:
#     targets = [ClassifierOutputTarget(vocab.word2idx[token])]
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
#     cv2.imwrite(f'heatmap_{token}.png', visualization[:,:,::-1])




# ----------------------------------------------------------------------------------
# image_path = "/scratch/acb11361bd/StoryGan/METER-Pororo-Single/data/15.png"
# input_tensor = get_image(image_path).unsqueeze(0)
# # print(input_tensor.size())
# rgb_image = cv2.imread(image_path)[:, :, ::-1]
# rgb_image = cv2.resize(rgb_image, (224, 224))
# rgb_image = np.float32(rgb_image) / 255



# for token in ['pororo', 'poby', 'eddy', 'crong']:
#     targets = [ClassifierOutputTarget(vocab.word2idx[token])]
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
#     cv2.imwrite(f'heatmap/heatmap_{token}.png', visualization[:,:,::-1])



# ----------------------------------------------------------------------------------
# image_path = "/scratch/acb11361bd/StoryGan/ResNet-GradCam/data/16458.png"
# input_tensor = get_image(image_path).unsqueeze(0)
# # print(input_tensor.size())
# rgb_image = cv2.imread(image_path)[:, :, ::-1]
# rgb_image = cv2.resize(rgb_image, (224, 224))
# rgb_image = np.float32(rgb_image) / 255
# for token in ['crong']:
#     targets = [ClassifierOutputTarget(vocab.word2idx[token])]
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
#     cv2.imwrite(f'heatmap_{token}.png', visualization[:,:,::-1])


# ----------------------------------------------------------------------------------
# image_path = "/scratch/acb11361bd/StoryGan/ResNet-GradCam/data/6912.png"
# input_tensor = get_image(image_path).unsqueeze(0)
# # print(input_tensor.size())
# rgb_image = cv2.imread(image_path)[:, :, ::-1]
# rgb_image = cv2.resize(rgb_image, (224, 224))
# rgb_image = np.float32(rgb_image) / 255
# for token in ['sculpture', 'crong']:
#     targets = [ClassifierOutputTarget(vocab.word2idx[token])]
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
#     cv2.imwrite(f'heatmap_{token}.png', visualization[:,:,::-1])



# ----------------------------------------------------------------------------------
# image_path = "/scratch/acb11361bd/StoryGan/GradCam-Pororo/data/29130.png"
# input_tensor = get_image(image_path).unsqueeze(0)
# # print(input_tensor.size())
# rgb_image = cv2.imread(image_path)[:, :, ::-1]
# rgb_image = cv2.resize(rgb_image, (224, 224))
# rgb_image = np.float32(rgb_image) / 255
# for token in ['bicycle', 'harry', 'crong', 'house']:
#     targets = [ClassifierOutputTarget(vocab.word2idx[token])]
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
#     cv2.imwrite(f'heatmap/heatmap_{token}.png', visualization[:,:,::-1])