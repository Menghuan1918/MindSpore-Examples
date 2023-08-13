import os
import pickle
import numpy as np
import mindspore
from mindspore.dataset import GeneratorDataset


class CIFAR10(object):
    train_list = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    ]

    test_list = [
        'test_batch',
    ]

    def __init__(self, root, train, transform=None, target_transform=None):
        super(CIFAR10, self).__init__()

        self.root = root
        self.train = train  # training set or test set
        
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
    
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

cifar10_test = CIFAR10(root="./cifar10/cifar-10-batches-py", train=False)
cifar10_test = GeneratorDataset(source=cifar10_test, column_names=["image", "label"])
cifar10_test = cifar10_test.batch(128)
for data in cifar10_test.create_dict_iterator():
    print(data["image"].shape, data["label"].shape)

(128, 32, 32, 3) (128,)
(128, 32, 32, 3) (128,)
(128, 32, 32, 3) (128,)
(128, 32, 32, 3) (128,)
