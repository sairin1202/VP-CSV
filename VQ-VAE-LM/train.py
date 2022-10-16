from test_dataset import VideoFolderDataset, StoryDataset, Vocabulary
import torchvision.transforms as transforms
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dalle_sequence_pytorch import DALLE
from torchvision.utils import save_image
from dalle_sequence_pytorch import VQGanVAE, CLIP
from dalle_sequence_pytorch import DiscreteVAE
import torch.nn as nn
import random

import PIL
import numpy as np
import os
import torch
import pickle
import nltk
from tqdm import tqdm

BATCH_SIZE = 96
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

vocab = Vocabulary(vocab_file='./vocab.pkl', annotations_file=None, vocab_threshold=None, vocab_from_file=True)
print('vocab length:', len(vocab))

model_path = "/scratch/acb11361bd/StoryGan/Pororo-VAE/taming/taming-transformers/logs/2021-12-22T13-14-01_custom_vqgan/testtube/version_0/checkpoints/epoch4.ckpt"
model_config = "/scratch/acb11361bd/StoryGan/Pororo-VAE/taming/taming-transformers/logs/2021-12-22T13-14-01_custom_vqgan/configs/2021-12-22T13-14-01-project.yaml"
save_path = "model/scratch/"



def get_model():
    vae = VQGanVAE(model_path, model_config)

    dalle = DALLE(
        dim = 768,
        vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
        num_text_tokens = len(vocab),    # vocab size for text
        text_seq_len = 360,         # text sequence length
        depth = 6,                 # should aim to be 64
        heads = 6,                 # attention heads
        dim_head = 64,              # attention head dimension
        attn_dropout = 0.1,         # attention dropout
        ff_dropout = 0.1            # feedforward dropout
    )

    dalle = dalle.cuda()
    return dalle



def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


def get_optimizer(model, lr, train_steps, warmup_steps):
    optimizer = AdamW(get_trainable_params(dalle), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    return optimizer, lr_scheduler


def get_dataloader():
    dir_path = "../StoryViz/data/"
    def video_transforms(video):
        image_transform = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((64, 64)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        vid = []
        for im in video:
            vid.append(image_transform(im))
        vid = torch.stack(vid).permute(1, 0, 2, 3)
        return vid

    video_len = 5
    n_channels = 3
    counter = np.load(os.path.join(dir_path, 'frames_counter.npy'), allow_pickle=True).item()

    base = VideoFolderDataset(dir_path, counter = counter, cache = dir_path, min_len = 4, mode="train")
    storydataset = StoryDataset(base, dir_path, video_transforms, return_caption=False)
    dataloader = torch.utils.data.DataLoader(storydataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=32)
    return dataloader





def tokenize(text):
    # print(len(vocab['word2idx']))
    tokens = nltk.tokenize.word_tokenize(text.lower())
    tokens = [vocab(tok) for tok in tokens]
    while len(tokens) < 72:
        tokens.append(1)
    tokens = torch.LongTensor(tokens[:72])
    mask = (tokens != 1).bool().cuda()
    return {'token': tokens, 'mask': mask}


def get_scheduler(opt):
    scheduler = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )
    return scheduler

if __name__ == "__main__":
    data = get_dataloader()
    dalle = get_model()
    dalle = nn.DataParallel(dalle)
    # clip = get_clip()
    # clip = nn.DataParallel(clip)
    epoch = 100
    batch_size = BATCH_SIZE
    # optimizer = get_optimizer(dalle)
    # scheduler = get_scheduler(optimizer)
    optimizer, scheduler = get_optimizer(model=dalle, lr=0.0008, train_steps=epoch*len(data), warmup_steps=0.1*epoch*len(data))


    for e in range(epoch):
        losses = []
        for idx, d in enumerate(tqdm(data)):
            batch_tokens = []
            batch_masks = []
            batch_img_seq = []
            for bs in range(batch_size):
                images = []
                descriptions = []
                masks = []
                for _ in range(5):
                    images.append(d['images'][bs][:,_,:,:])
                    text = d['text'][_][bs]
                    descriptions.append(tokenize(text)['token'])
                    masks.append(tokenize(text)['mask'])
                # print(descriptions)
                # random.shuffle(descriptions)
                # print(descriptions)
                tokens = torch.cat(descriptions)
                # print(tokens.size())
                masks = torch.cat(masks)
                
                img_seq = torch.stack(images)
                batch_img_seq.append(img_seq)
                batch_masks.append(masks)
                batch_tokens.append(tokens)

            batch_img_seq = torch.stack(batch_img_seq)
            batch_tokens = torch.stack(batch_tokens)
            batch_masks = torch.stack(batch_masks)
            # print(batch_img_seq.size(), batch_tokens.size(), batch_masks.size())
            optimizer.zero_grad()
            loss = dalle(text=batch_tokens, image=batch_img_seq, mask = batch_masks, return_loss = True)
            loss = loss.mean()
            # print(loss)
            loss.backward()
            # clip_grad_norm_(dalle.parameters(), 0.5)
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()
        # scheduler.step(torch.mean(torch.Tensor(losses)))
        print(f'Epoch {e} loss:', np.mean(losses))
        if e % 5 == 0 and e >= 30:
            torch.save(dalle.module.state_dict(), save_path + f'{e}.pth')
        