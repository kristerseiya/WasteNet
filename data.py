
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import os
import numpy as np

transform_train = transforms.Compose([transforms.RandomResizedCrop(200),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(180),
                                      transforms.ColorJitter(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

transform_infer = transforms.Compose([transforms.Resize(200),
                                      transforms.CenterCrop(200),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

def get_label(file_path):
    dir = file_path.split('/')[-2]
    if dir in ['cardboard_me', 'cardboard_trashnet', 'paper_box_google',
               'paper_box_me', 'paper_bag_me', 'other_paper_me']:
        return 0
    elif dir in ['dirty_plate_food_google', 'food_contamination_me', 'food_me']:
        return 1
    elif dir in ['disposable_cup_google', 'disposable_cup_me']:
        return 2
    elif dir in ['glass_me', 'glass_someone']:
        return 3
    elif dir in ['metal_can_someone', 'metal_me']:
        return 4
    # elif dir in ['paper_bag_me']:
    #     return 5
    elif dir in ['plastic_bag_google', 'plastic_bag_wrap_me']:
        return 5
    elif dir in ['plastic_bottle_google', 'plastic_bottle_in_hand_google',
                 'plastic_bottle_me', 'plastic_bottle_someone']:
        return 6
    elif dir in ['other_plastic_me']:
        return 7
    elif dir in ['snack_wrappers_google', 'food_wrap_me']:
        return 8
    elif dir in ['battery_me', 'electronic_me', 'napkin_me', 'other_me',
                 'pizza_box_google', 'pizza_box_me',
                 'styrofoam_google', 'styrofoam_me']:
         return 9
    return -1

def id_label(label):
    classes = ['paper', 'food contamination', 'disposable cup',
               'glass', 'metal', 'plastic bag', 'plastic bottle',
               'other plastic', 'food / snack wraps', 'others']
    return classes[label]

class WasteNetSubset(Dataset):
    def __init__(self, dataset, indices, mode):
        self.dataset = dataset
        self.indices = indices

        if mode == 'train':
            self.transform = transform_train
        elif mode in ['val', 'test']:
            self.transform = transform_infer

    def __getitem__(self, idx):
        imgs, labels = self.dataset[self.indices[idx]]
        return self.transform(imgs), labels

    def __len__(self):
        return len(self.indices)

    def print_stats(self):
        (unique, counts) = np.unique(self.dataset.labels[self.indices], return_counts=True)
        for u, c in zip(unique, counts):
            print('{:s}: {:d}'.format(id_label(u), c))

class WasteNetDataset(Dataset):
    def __init__(self, root_dir, mode):
        super().__init__()
        self.images = []
        self.labels = []

        for file_path in glob.glob(os.path.join(root_dir, '*/**.png')):
            label = get_label(file_path)
            if label != -1:
                fptr = Image.open(file_path)
                file_copy = fptr.copy()
                fptr.close()
                self.images.append(file_copy)
                self.labels.append(label)

        if mode == 'train':
            self.transform = transform_train
        elif mode in ['val', 'test']:
            self.transform = transform_infer
        elif mode == 'none':
            self.transform = lambda x: x

    def set_mode(self, mode):
        if mode == 'train':
            self.transform = transform_train
        elif mode in ['val', 'test']:
            self.transform = transform_infer
        elif mode == 'none':
            self.transform = lambda x: x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]

    def print_stats(self):
        (unique, counts) = np.unique(self.labels, return_counts=True)
        for u, c in zip(unique, counts):
            print('{:s}: {:d}'.format(id_label(u), c))

    def split(self, ratios):
        ratios = ratios / np.sum(ratios)
        total_num = len(self.labels)
        indices = list(range(total_num))
        np.random.shuffle(indices)

        split1 = int(np.floor(total_num * ratios[0]))
        split2 = int(np.floor(total_num * ratios[1]))

        train_set = WasteNetSubset(self, indices[:split1], 'train')
        val_set = WasteNetSubset(self, indices[split1:split1+split2], 'val')
        test_set = WasteNetSubset(self, indices[split1+split2:], 'test')

        return train_set, val_set, test_set
