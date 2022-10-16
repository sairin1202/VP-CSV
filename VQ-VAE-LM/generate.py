from dataset import StoryDataset, Vocabulary
from tqdm import tqdm
import torchvision.transforms as transforms
import PIL
import numpy as np
import os
import torch
import pickle
import nltk

from dalle_sequence_pytorch import DALLE
from torchvision.utils import save_image
from dalle_sequence_pytorch import VQGanVAE
from dalle_sequence_pytorch import DiscreteVAE, CLIP
import argparse

vocab = Vocabulary(vocab_file='./vocab.pkl', annotations_file=None, vocab_threshold=None, vocab_from_file=True)

vae_path = "/scratch/acb11361bd/StoryGan/Pororo-VAE/taming/taming-transformers/logs/2021-12-22T13-14-01_custom_vqgan/testtube/version_0/checkpoints/epoch4.ckpt"
vae_config = "/scratch/acb11361bd/StoryGan/Pororo-VAE/taming/taming-transformers/logs/2021-12-22T13-14-01_custom_vqgan/configs/2021-12-22T13-14-01-project.yaml"


BATCH_SIZE=32
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
        dataloader = torch.utils.data.DataLoader(storydataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=8)
    else:
        dataloader = torch.utils.data.DataLoader(storydataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=8)
    return dataloader


def get_model():

    vae = VQGanVAE(vae_path, vae_config)

    # dalle = DALLE(
    #     dim = 1024,
    #     vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
    #     num_text_tokens = len(vocab),    # vocab size for text
    #     text_seq_len = 360,         # text sequence length
    #     depth = 12,                 # should aim to be 64
    #     heads = 8,                 # attention heads
    #     dim_head = 64,              # attention head dimension
    #     attn_dropout = 0.1,         # attention dropout
    #     ff_dropout = 0.1            # feedforward dropout
    # )
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





def single_test(dalle, clip, model_path):
    ep = 30
    dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
    dalle.eval()
    # texts = [ "crong turns his face to pororo on the ladder.",
    #         "Crong smiles and Pororo is closing his eyes and saying something to crong.",
    #         "Pororo is angry and says to crong with a angry voice. crong is listening what pororo says.",
    #         "Crong is turns his body and just draws a picture on the wall.",
    #         "Pororo shakes the ladder and crong looses his balance and crong is about to fall down on the ladder."]
    # texts = [ "Crong smiles and Pororo is closing his eyes and saying something to crong.",
    #         "Pororo is angry and says to crong with a angry voice. crong is listening what pororo says.",
    #         "Crong is turns his body and just draws a picture on the wall.",
    #         "Pororo shakes the ladder and crong looses his balance and crong is about to fall down on the ladder.",
    #         "crong falls down. and rubs his head. Pororo loses his tempo."]
    texts = ["Crong falls down and loses his tempo.", 
            "Crong smiles and Pororo is closing his eyes and saying something to crong.",
            "Pororo is angry and says to crong with a angry voice. crong is listening what pororo says.",
            "Crong is turns his body and just draws a picture on the wall.",
            "Pororo shakes the ladder and crong looses his balance and crong is about to fall down on the ladder.",
            ]    
    tokens = []
    masks = []
    for _ in range(len(texts)):
        token = tokenize(texts[_])['token']
        mask = tokenize(texts[_])['mask']
        tokens.append(token)
        masks.append(mask)
    tokens = torch.cat(tokens, dim=-1)
    masks = torch.cat(masks, dim=-1)
    
    print('start generating image...')
    
    # best_images, best_loss = None, -1e10
    # images, loss = dalle.generate_images(tokens, mask = masks, clip=clip, filter_thres=0.8)
    sample_images = []
    for _ in range(5):

        images = dalle.generate_images(tokens, mask = masks, filter_thres=1)
        images = torch.cat(images, dim=-1)
        sample_images.append(images)

    save_image(torch.cat(sample_images, dim=-2), f'res/sample/sample3.png', normalize=True)
    # best_images, best_loss = None, -1e10
    # for j, l in enumerate(loss):
    #     if l.item() > best_loss:
    #         best_loss = l.item()
    #         best_images = images[j]
    # # images = dalle.generate_images(texts, mask = masks, filter_thres=0.5)
    # save_image(best_images, f'res/sample/img_ep_pre{ep}{suffix}_best.png', normalize=True)

def generate(dalle, clip, batch_size, save_path, filter_thres):
    cnt = 0
    data = get_dataloader(train=False)
    for idx, d in enumerate(tqdm(data)):
        batch_tokens = []
        batch_masks = []
        for bs in range(len(d['text'][0])):
            descriptions = []
            masks = []
            for _ in range(5):
                text = d['text'][_][bs]
                descriptions.append(tokenize(text)['token'][0])
                masks.append(tokenize(text)['mask'][0])
            # print(descriptions[0].size())
            tokens = torch.cat(descriptions)
            # print(tokens.size())
            masks = torch.cat(masks)
            batch_masks.append(masks)
            batch_tokens.append(tokens)
 
        batch_tokens = torch.stack(batch_tokens)
        batch_masks = torch.stack(batch_masks)
        images =  dalle.generate_images(batch_tokens, mask = batch_masks, filter_thres=filter_thres, temperature=0.7)

        for _ in range(len(images)):
            for i in range(len(images[_])):
                if not os.path.exists(f'res/{save_path}'):
                    os.system(f'mkdir res/{save_path}')
                save_image(images[_][i], f'res/{save_path}/img-{i+cnt}-{_}.png', normalize=True)
        cnt += batch_size




def get_test_image():
    cnt = 0
    data = get_dataloader(train=False)
    for idx, d in enumerate(tqdm(data)):
        batch_tokens = []
        batch_masks = []
        batch_img_seq = []
        for bs in range(len(d['text'][0])):
            descriptions = []
            masks = []
            images = []
            for _ in range(5):
                # print(d['images'][bs][:,_,:,:].size())
                # print(d['text'])
                save_image(d['images'][bs][:,_,:,:], f'res/test/img-{cnt}-{_}.png', normalize=False)
            cnt += 1



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
            sample_captions.append(f'----------------{bs}------------------\n')
            tokens = torch.cat(descriptions)
            masks = torch.cat(masks)
            batch_masks.append(masks)
            batch_tokens.append(tokens)

        batch_gold_images = torch.cat(batch_gold_images, dim=-2).cuda()
        batch_tokens = torch.stack(batch_tokens)
        batch_masks = torch.stack(batch_masks)
        images =  dalle.generate_images(batch_tokens, mask = batch_masks, filter_thres=1)
        images = torch.cat(images, dim=-1)
        images = [im for im in images]
        images = torch.cat(images, dim=-2)
        sample_image = torch.cat([images, torch.zeros(3, 64*batch_size, 64).cuda(), batch_gold_images], dim=-1)
        cnt += 1
        with open(save_path+f'{cnt}.txt', 'w') as f:
            f.writelines(sample_captions)
        save_image(sample_image, save_path+f'{cnt}.png', normalize=False)



clip = None
dalle = get_model().cuda()
dalle.eval()

# get_test_image()
# model_path = "model/vae_dataloader/"
# single_test(dalle, clip, model_path)



# model_path = "model/finetune/"
# ep = 80
# dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
# dalle.eval()
# batch_size = BATCH_SIZE
# save_path = f'finetune_ep{ep}_gen_filter_1'
# generate(dalle, clip, batch_size, save_path, 1)


model_path = "model/finetune/"
ep = 80
dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
dalle.eval()
batch_size = BATCH_SIZE
save_path = f'res/sample/finetune_ep{ep}/'
if not os.path.exists(f'res/sample/finetune_ep{ep}'):
    os.system(f'mkdir -p ./res/sample/finetune_ep{ep}')
generate_images(dalle, batch_size, save_path)


# model_path = "model/scratch/"
# ep = 50
# dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
# dalle.eval()
# batch_size = BATCH_SIZE
# save_path = f'scratch_ep{ep}_gen_filter_0_99'
# generate(dalle, clip, batch_size, save_path, 0.99)

# model_path = "model/scratch/"
# ep = 60
# dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
# dalle.eval()
# batch_size = BATCH_SIZE
# save_path = f'scratch_ep{ep}_gen_filter_0_9'
# generate(dalle, clip, batch_size, save_path, 0.95)





# model_path = "model/vae_scratch/"
# # get_test_image()
# for ep in [70]:
#     dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
#     dalle.eval()
#     batch_size = 8
#     save_path = f'vae_scratch_ep{ep}_gen_filter_0_9'
#     generate(dalle, clip, batch_size, save_path)

# model_path = "model/vae_pretrain_mask3/"
# for ep in [195]:
#     dalle.load_state_dict(torch.load(model_path + f'{ep}.pth'))
#     dalle.eval()
#     batch_size = 8
#     save_path = f'res/sample/vae_pretrain_mask3_dalle_ep{ep}_gen_filter_1/'
#     if not os.path.exists(f'res/sample/vae_pretrain_mask3_dalle_ep{ep}_gen_filter_1'):
#         os.system(f'mkdir -p ./res/sample/vae_pretrain_mask3_dalle_ep{ep}_gen_filter_1')
#     generate_all(dalle, clip, save_path)



