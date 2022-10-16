from dataset import StoryDataset, Vocabulary
from tqdm import tqdm
import torchvision.transforms as transforms
import PIL
import numpy as np
import os
import torch
import pickle
import nltk
import torch.nn as nn

from dalle import DALLE
from torchvision.utils import save_image
from dalle import VQGanVAE
from dalle import DiscreteVAE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vocab = Vocabulary(vocab_file='./vocab.pkl', annotations_file=None, vocab_threshold=None, vocab_from_file=True)

BATCH_SIZE = 32



vae_path = "/scratch/acb11361bd/StoryGan/Pororo-VAE/taming/taming-transformers/logs/2021-12-22T13-14-01_custom_vqgan/testtube/version_0/checkpoints/epoch4.ckpt"
vae_config = "/scratch/acb11361bd/StoryGan/Pororo-VAE/taming/taming-transformers/logs/2021-12-22T13-14-01_custom_vqgan/configs/2021-12-22T13-14-01-project.yaml"



import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--step_start', type=int, default=0)
parser.add_argument('--step_end', type=int, default=0)

args = parser.parse_args()
step_start = args.step_start
step_end = args.step_end


def resize_layout(layouts):
    bs = layouts.size(0)
    batch_resized_layout = []
    for b in range(bs):
        resized_layout = []
        for _ in range(5):
            resized_tensor = [[0]*64 for _ in range(64)]
            for i in range(len(resized_tensor)):
                for j in range(len(resized_tensor)):
                    resized_tensor[i][j] = layouts[b][_][i//8][j//8]
            resized_layout.append(torch.Tensor(resized_tensor))
        resized_layout = torch.stack(resized_layout)
        batch_resized_layout.append(resized_layout)
    return torch.stack(batch_resized_layout)

def get_dataloader(train):
    data_dir = "./data/"
    def video_transforms(video):
        image_transform = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((64, 64)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        vid = []
        for im in video:
            # print(im.shape)
            vid.append(image_transform(im))
        vid = torch.stack(vid).permute(1, 0, 2, 3)
        return vid

    video_len = 5
    n_channels = 3
    storydataset = StoryDataset(data_dir, video_transforms, train=train)
    if train:
        dataloader = torch.utils.data.DataLoader(storydataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=32)
    else:
        dataloader = torch.utils.data.DataLoader(storydataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=32)
    return dataloader



def get_model():
    # vae = OpenAIDiscreteVAE()
    vae = VQGanVAE(vae_path, vae_config)

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




def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    tokens = [vocab(tok) for tok in tokens]
    while len(tokens) < 72:
        tokens.append(1)
    tokens = torch.LongTensor(tokens[:72]).cuda()
    mask = (tokens != 1).bool().cuda()
    tokens = torch.stack([tokens], dim=0)
    mask = torch.stack([mask], dim=0)
    return {'token': tokens, 'mask': mask}



 

def single_test(dalle1, dalle2, clip):

    texts = ["loopy asks whether it is because of cookies.", "loopy denies with his hands.", "Loopy hands her cookies to Eddy.", "Loopy gives her cookies.", " Crong sighs."]
    tokens = []
    masks = []
    for _ in range(len(texts)):
        token = tokenize(texts[_])['token']
        mask = tokenize(texts[_])['mask']
        tokens.append(token)
        masks.append(mask)

    tokens = torch.cat(tokens, dim=-1)
    print(tokens)
    masks = torch.cat(masks, dim=-1)
    batch_tokens = tokens.repeat(1, 1)
    batch_masks = masks.repeat(1, 1)
    print(tokens.size(), masks.size())
    print('start generating image...')
    
    # best_images, best_loss = None, -1e10
    # images, loss = dalle.generate_images(tokens, mask = masks, clip=clip, filter_thres=0.8)
    layout =  dalle1.fill_layout(batch_tokens, layout = torch.zeros(batch_tokens.size(0), 320).long().cuda(), mask = batch_masks, filter_thres=1)
    layout01 = layout.clone()
    layout01[layout01>0] = 1
    
    print('after fill', layout[0][:100])
    images =  dalle2.generate_images(batch_tokens, layout_fill = layout, mask = batch_masks, filter_thres=1)
    # revise layout images
    layout01 = layout01.view(-1, 5, 8, 8)
    layout01 = resize_layout(layout01).cuda()
    layout01 = torch.transpose(layout01, 0, 1)
    layout01 = layout01.unsqueeze(2)
    layout01 = layout01.repeat(1,1,3,1,1)

    images = torch.cat(images, dim=-1)
    images = [im for im in images]
    images = torch.cat(images, dim=-2)


    layout01 = torch.cat([l for l in layout01], dim=-1)
    layout01 = [im for im in layout01]
    layout01 = torch.cat(layout01, dim=-2)

    mask_images = images * layout01

    sample_image = torch.cat([mask_images, torch.zeros(3, 64, 64).cuda(), images], dim=-1)

    save_image(sample_image, f'res/sample/sample.png', normalize=True)


def generate(dalle, batch_size, save_path, filter_thres):
    cnt = 0
    data = get_dataloader(train=False)
    
    for idx, d in enumerate(tqdm(data)):
        batch_tokens = []
        batch_masks = []
        batch_gold_images = []
        sample_images = []
        sample_captions = []
        for bs in range(len(d['text'][0])):
            descriptions = []
            masks = []
            gold_images = []
            for _ in range(5):
                sample_captions.append(d['text'][_][bs]+'\n')
                text = d['text'][_][bs]
                descriptions.append(tokenize(text)['token'][0])
                masks.append(tokenize(text)['mask'][0])
                gold_images.append(d['images'][bs][:,_,:,:])
            batch_gold_images.append(torch.cat(gold_images, dim=-1))
            tokens = torch.cat(descriptions)
            masks = torch.cat(masks)
            batch_masks.append(masks)
            batch_tokens.append(tokens)
        batch_gold_images = torch.cat(batch_gold_images, dim=-2).cuda()
        batch_tokens = torch.stack(batch_tokens).cuda()
        batch_masks = torch.stack(batch_masks).cuda()
        # data parallel module
        layout_seq, images =  dalle.generate_images(batch_tokens, mask = batch_masks, filter_thres=filter_thres, temperature=0.7)

        for _ in range(len(images)):
            for i in range(len(images[_])):
                if not os.path.exists(f'{save_path}'):
                    os.system(f'mkdir {save_path}')
                save_image(images[_][i], f'{save_path}/img-{i+cnt}-{_}.png', normalize=True)
        cnt += batch_size




def generate_images(dalle, batch_size, save_path):
    cnt = 0
    data = get_dataloader(train=False)
    
    for idx, d in enumerate(tqdm(data)):
        batch_tokens = []
        batch_masks = []
        batch_gold_images = []
        sample_images = []
        sample_captions = []
        for bs in range(len(d['text'][0])):
            descriptions = []
            masks = []
            gold_images = []
            for _ in range(5):
                sample_captions.append(d['text'][_][bs]+'\n')
                text = d['text'][_][bs]
                descriptions.append(tokenize(text)['token'][0])
                masks.append(tokenize(text)['mask'][0])
                gold_images.append(d['images'][bs][:,_,:,:])
            batch_gold_images.append(torch.cat(gold_images, dim=-1))
            tokens = torch.cat(descriptions)
            masks = torch.cat(masks)
            batch_masks.append(masks)
            batch_tokens.append(tokens)
            sample_captions.append(f'----------------{bs}------------------\n')
        batch_gold_images = torch.cat(batch_gold_images, dim=-2).cuda()
        batch_tokens = torch.stack(batch_tokens).cuda()
        batch_masks = torch.stack(batch_masks).cuda()

        layout, images =  dalle.generate_images(batch_tokens, mask = batch_masks, filter_thres=0.99, temperature=0.6)
        print('layout', layout.size())
        layout01 = layout.clone()
        layout01[layout01>0] = 1
        
        print('after fill', layout[0][:100])
        layout01 = layout01.view(-1, 5, 8, 8)
        print(layout01.size())
        layout01 = resize_layout(layout01).cuda()
        layout01 = torch.transpose(layout01, 0, 1)
        layout01 = layout01.unsqueeze(2)
        layout01 = layout01.repeat(1,1,3,1,1)

        images = torch.cat(images, dim=-1)
        images = [im for im in images]
        images = torch.cat(images, dim=-2)


        layout01 = torch.cat([l for l in layout01], dim=-1)
        layout01 = [im for im in layout01]
        layout01 = torch.cat(layout01, dim=-2)
        print(images.size(), layout01.size())
        mask_images = images * layout01

        sample_image = torch.cat([mask_images, torch.zeros(3, batch_size*64, 64).cuda(), images, torch.zeros(3, batch_size*64, 64).cuda(), batch_gold_images], dim=-1)
        cnt += 1
        with open(save_path+f'{cnt}.txt', 'w') as f:
            f.writelines(sample_captions)
        save_image(sample_image, save_path+f'{cnt}.png', normalize=False)



def generate_gold_images(dalle1, dalle2, clip, batch_size, save_path):
 
    cnt = 0
    data = get_dataloader(train=False)
    
    for idx, d in enumerate(tqdm(data)):
        batch_tokens = []
        batch_masks = []
        batch_gold_images = []
        batch_layouts = []
        batch_imgs = []
        sample_images = []
        sample_captions = []
        for bs in range(len(d['text'][0])):
            descriptions = []
            masks = []
            gold_images = []
            images = []
            for _ in range(5):
                sample_captions.append(d['text'][_][bs]+'\n')
                text = d['text'][_][bs]
                images.append(d['images'][bs][:,_,:,:])
                descriptions.append(tokenize(text)['token'][0])
                masks.append(tokenize(text)['mask'][0])
                gold_images.append(d['images'][bs][:,_,:,:])
                

            imgs = torch.stack(images)
            batch_gold_images.append(torch.cat(gold_images, dim=-1))
            tokens = torch.cat(descriptions)
            masks = torch.cat(masks)
            batch_masks.append(masks)
            batch_tokens.append(tokens)
            sample_captions.append(f'----------------{bs}------------------\n')
            batch_imgs.append(imgs)

        batch_imgs = torch.stack(batch_imgs)
        batch_gold_images = torch.cat(batch_gold_images, dim=-2).cuda()
        batch_tokens = torch.stack(batch_tokens).cuda()
        batch_masks = torch.stack(batch_masks).cuda()
        layout =  d['image_layout'].cuda()

        imgs = []
        for _ in range(5):
            img = batch_imgs[:,_,:,:,:].cuda()
            img = dalle2.vae.get_codebook_indices(img) + 1
            imgs.append(img)

        imgs = torch.cat(imgs, dim=-1).cuda()
        layout = imgs * layout


        layout01 = layout.clone()
        layout01[layout01>0] = 1
        
        print('after fill', layout[0][:100])
        images =  dalle2.generate_images(batch_tokens, layout_fill = layout, mask = batch_masks, filter_thres=1)
        # revise layout images
        layout01 = layout01.view(-1, 5, 8, 8)
        layout01 = resize_layout(layout01).cuda()
        layout01 = torch.transpose(layout01, 0, 1)
        layout01 = layout01.unsqueeze(2)
        layout01 = layout01.repeat(1,1,3,1,1)

        images = torch.cat(images, dim=-1)
        images = [im for im in images]
        images = torch.cat(images, dim=-2)


        layout01 = torch.cat([l for l in layout01], dim=-1)
        layout01 = [im for im in layout01]
        layout01 = torch.cat(layout01, dim=-2)

        mask_images = images * layout01

        sample_image = torch.cat([mask_images, torch.zeros(3, batch_size*64, 64).cuda(), images, torch.zeros(3, batch_size*64, 64).cuda(), batch_gold_images], dim=-1)
        cnt += 1
        with open(save_path+f'{cnt}.txt', 'w') as f:
            f.writelines(sample_captions)
        save_image(sample_image, save_path+f'{cnt}.png', normalize=False)



dalle = get_model()
dalle = dalle.cuda()

# ep = 10
# model_path = "model/semantic/"
# dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
# dalle.eval()
# batch_size = BATCH_SIZE
# save_path = f'res/semantic_ep{ep}_gen_filter_0_99/'
# if not os.path.exists(f'res/semantic_ep{ep}_gen_filter_0_99'):
#     os.system(f'mkdir -p ./res/semantic_ep{ep}_gen_filter_0_99')
# generate(dalle, batch_size, save_path, 0.99)



# ep = 50
# model_path = "model/finetune/"
# dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
# dalle.eval()
# batch_size = BATCH_SIZE
# save_path = f'res/finetune_ep{ep}_gen_filter_0_99/'
# if not os.path.exists(f'res/finetune_ep{ep}_gen_filter_0_99'):
#     os.system(f'mkdir -p ./res/finetune_ep{ep}_gen_filter_0_99')
# generate(dalle, batch_size, save_path, 0.99)

# ep =50
# model_path = "model/scratch/"
# dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
# dalle.eval()
# batch_size = BATCH_SIZE
# save_path = f'res/scratch_ep{ep}_gen_filter_0_99/'
# if not os.path.exists(f'res/scratch_ep{ep}_gen_filter_0_99'):
#     os.system(f'mkdir -p ./res/scratch_ep{ep}_gen_filter_0_99')
# generate(dalle, batch_size, save_path, 0.99)





ep = 45
model_path = "model/semantic/"
dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
dalle.eval()
batch_size = BATCH_SIZE
save_path = f'res/sample/semantic_ep{ep}/'
if not os.path.exists(f'res/sample/semantic_ep{ep}'):
    os.system(f'mkdir -p ./res/sample/semantic_ep{ep}')
generate_images(dalle, batch_size, save_path)

