import torchvision.transforms as transforms
import argparse
import os
from vfid.fid_score import fid_score
from dataset import StoryImageTestDataset

def fid(img_gen_dir, img_ref_dir, imsize):
    image_transforms = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    ref_dataset = StoryImageTestDataset(img_ref_dir,
                                    imsize,
                                    mode='test',
                                    out_img_folder=img_test_dir,
                                    transform=image_transforms)
    gen_dataset = StoryImageTestDataset(img_ref_dir,
                                    imsize,
                                    mode='test',
                                    out_img_folder=img_gen_dir,
                                    transform=image_transforms)



    # os.path.join(img_ref_dir, 'fid_cache_test.npz')
    fid = fid_score(ref_dataset, gen_dataset, cuda=True, normalize=True, r_cache='/scratch/acb11361bd/StoryGan/StoryViz/data/fid_cache_test.npz', batch_size=1)
    print('Frechet Image Distance: ', fid)

if __name__ == "__main__":

    imsize = 64
    img_test_dir = '/scratch/acb11361bd/StoryGan/PlanAndDraw-Pororo-Pretrain-stage2-mask/res/test/'
    img_ref_dir = '../StoryViz/data/'

    img_gen_dir = '/scratch/acb11361bd/StoryGan/PlanAndDraw-Pororo-stage2-semantic/res/scratch_ep20_gen_filter_0_99'
    print(img_gen_dir)
    fid(img_gen_dir, img_ref_dir, imsize)

