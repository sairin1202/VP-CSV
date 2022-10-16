import os, pickle, re, csv
from tqdm import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import PIL
from collections import Counter
import nltk
import json
import random


class Vocabulary(object):

    def __init__(self,
                 vocab_threshold,
                 vocab_file,
                 annotations_file,
                 vocab_from_file=False,
                 unk_word="[UNK]",
                 pad_word="[PAD]",
                 start_word="[BOS]",
                 end_word="[EOS]"):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.start_word=start_word
        self.end_word = end_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            print('Reading vocabulary from %s file!' % self.vocab_file)
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab['word2idx']
                self.idx2word = vocab['idx2word']
            print('Vocabulary successfully loaded from %s file!' % self.vocab_file)
        else:
            print("Building voabulary from scratch")
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word('[MASK]')
        self.add_word('[UNMASK]')
        self.add_word('[SEP]')
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)
        self.add_captions()
        

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        counter = Counter()
        with open(self.annotations_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print("Tokenizing captions")
            for i, row in tqdm(enumerate(csv_reader)):
                _, _, caption = row
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def load_glove(self, filename):
        """ returns { word (str) : vector_embedding (torch.FloatTensor) }
        """
        glove = {}
        with open(filename) as f:
            for line in tqdm(f.readlines()):
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def extract_glove(self, raw_glove_path, vocab_glove_path, glove_dim=300):

        if os.path.exists(vocab_glove_path):
            print("Pre-extracted embedding matrix exists at %s" % vocab_glove_path)
        else:
            # Make glove embedding.
            print("Loading glove embedding at path : {}.\n".format(raw_glove_path))
            glove_full = self.load_glove(raw_glove_path)
            print("Glove Loaded, building word2idx, idx2word mapping.\n")
            idx2word = {v: k for k, v in self.word2idx.items()}

            glove_matrix = np.zeros([len(self.word2idx), glove_dim])
            glove_keys = glove_full.keys()
            for i in tqdm(range(len(idx2word))):
                w = idx2word[i]
                w_embed = glove_full[w] if w in glove_keys else np.random.randn(glove_dim) * 0.4
                glove_matrix[i, :] = w_embed
            print("vocab embedding size is :", glove_matrix.shape)
            torch.save(glove_matrix, vocab_glove_path)

    def __call__(self, word):
        if not word in self.word2idx:
            # print(word, 'not in vocab')
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)




class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform, train):
        self.transforms = transform
        self.data_dir = data_dir 
        if train:
            with open('./data/train_story.pkl', 'rb') as f:
                self.data = pickle.load(f)
            self.data_dir += 'train/'
        else:
            with open('./data/test_story.pkl', 'rb') as f:
                self.data = pickle.load(f)
            self.data_dir += 'test/'

        with open('data/image_index.pkl','rb') as f:
            self.image_index = pickle.load(f)

        with open('./data/test.pkl', 'rb') as f:
            self.gold_layout = pickle.load(f)
            # print(self.gold_layout)


    def __getitem__(self, item):
        story = self.data[item]['text']
        images = self.data[item]['image']
        text = []
        image = []
        img_index = []
        for idx, im_names in enumerate(images):
            img_path = str(im_names) 
            im = PIL.Image.open(img_path).convert('RGB')
            image.append(np.array(im))
            text.append(story[idx])
            img_index.append(torch.LongTensor(self.gold_layout[img_path.split('/')[-1][:-4]]))
        img_index = torch.cat(img_index)

        image = self.transforms(image)  
        data_item = {'images': image, 'text':text, 'image_names': images, 'image_layout': img_index}
        return data_item

    def __len__(self):
        return len(self.data)


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.transforms = transform
        self.data_dir = data_dir 
        with open('./data/new_pretrain_story.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, item):
        story = self.data[item]['text']
        images = self.data[item]['image']
        text = []
        image = []
        for idx, im_names in enumerate(images):
            if 'data' in im_names:
                im_names = "/scratch/acb11361bd/StoryGan/PlanAndDraw-Pororo-Pretrain-mask/" + im_names
            img_path = str(im_names) 
            im = PIL.Image.open(img_path).convert('RGB')
            image.append(np.array(im))
            text.append(story[idx])
        image = self.transforms(image)

        im_index = []
        for _ in range(5):
            rand_mask = [0]*64
            for _ in range(32):
                rand_int = random.randint(0, 63)
                rand_mask[rand_int] = 1
            im_index.append(rand_mask)
        im_index = np.concatenate(im_index, axis = 0)

        data_item = {'images': image, 'text':text, 'image_names': images, 'image_layout': torch.Tensor(im_index).long()}
        return data_item

    def __len__(self):
        return 300000
        

class StoryImageTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, im_input_size,
                 out_img_folder = '/ssd-playpen/home/adyasha/projects/StoryGAN/pororo_code_mod/output/pororo_both_stageI_r1.0/Test/images-epoch-110/',
                 mode='train',
                 video_len = 5,
                 transform=None):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > self.counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        # train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)

        if mode == 'train':
            self.ids = train_ids
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(im_input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            if mode == 'val':
                self.ids = val_ids[:2304]
            elif mode == 'test':
                self.ids = test_ids
            else:
                raise ValueError

            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    transforms.CenterCrop(im_input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        self.out_dir = out_img_folder

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        img_id = self.ids[item]
        img_paths = [str(self.images[img_id])[2:-1]] + [str(self.followings[img_id][k])[2:-1] for k in range(0, self.video_len-1)]
        if self.out_dir is not None:
            images = [PIL.Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, k))).convert('RGB') for k in range(self.video_len)]
        else:
            images = [self.sample_image(PIL.Image.open(os.path.join(self.img_folder, path)).convert('RGB')) for path in img_paths]
        labels = [self.labels[path.replace('.png', '').replace(self.img_folder + '/', '')] for path in img_paths]
        # return torch.cat([self.transform(image).unsqueeze(0) for image in images], dim=0), torch.tensor(np.vstack(labels))
        return torch.stack([self.transform(im) for im in images]), torch.tensor(np.vstack(labels))

    def __len__(self):
        return len(self.ids)






class StoryImageTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, im_input_size,
                 out_img_folder = '/ssd-playpen/home/adyasha/projects/StoryGAN/pororo_code_mod/output/pororo_both_stageI_r1.0/Test/images-epoch-110/',
                 mode='train',
                 video_len = 5,
                 transform=None):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > self.counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        # train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_val_test_ids.npy'), allow_pickle=True)
        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)

        if mode == 'train':
            self.ids = train_ids
            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(im_input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            if mode == 'val':
                self.ids = val_ids[:2304]
            elif mode == 'test':
                self.ids = test_ids
            else:
                raise ValueError

            if transform:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    transforms.CenterCrop(im_input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        self.out_dir = out_img_folder

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        img_id = self.ids[item]
        img_paths = [str(self.images[img_id])[2:-1]] + [str(self.followings[img_id][k])[2:-1] for k in range(0, self.video_len-1)]
        if self.out_dir is not None:
            images = [PIL.Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, k))).convert('RGB') for k in range(self.video_len)]
        else:
            images = [self.sample_image(PIL.Image.open(os.path.join(self.img_folder, path)).convert('RGB')) for path in img_paths]
        labels = [self.labels[path.replace('.png', '').replace(self.img_folder + '/', '')] for path in img_paths]
        # return torch.cat([self.transform(image).unsqueeze(0) for image in images], dim=0), torch.tensor(np.vstack(labels))
        return torch.stack([self.transform(im) for im in images]), torch.tensor(np.vstack(labels))

    def __len__(self):
        return len(self.ids)