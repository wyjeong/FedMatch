import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf

from config import *
from misc.utils import *
from torchvision import datasets,transforms

class DataGenerator:

    def __init__(self, args):
        """ Data Generator

        Generates batch-iid and batch-non-iid under both
        labels-at-client and labels-at-server scenarios.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """

        self.args = args
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task) 
        self.shape = (32,32,3)

    def generate_data(self):
        print('generating {} ...'.format(self.args.task))
        start_time = time.time()
        self.task_cnt = -1
        self.is_labels_at_server = True if 'server' in self.args.scenario else False
        self.is_imbalanced = True if 'imb' in self.args.task else False
        x, y = self.load_dataset(self.args.dataset_id)
        self.generate_task(x, y)
        print(f'{self.args.task} done ({time.time()-start_time}s)')

    def load_dataset(self, dataset_id):
        temp = {}
        if self.args.dataset_id_to_name[dataset_id] == 'cifar_10':
            temp['train'] = datasets.CIFAR10(self.args.dataset_path, train=True, download=True) 
            temp['test'] = datasets.CIFAR10(self.args.dataset_path, train=False, download=True) 
            x, y = [], []
            for dtype in ['train', 'test']:
                for image, target in temp[dtype]:
                    x.append(np.array(image))
                    y.append(target)

        x, y = self.shuffle(x, y)
        print(f'{self.args.dataset_id_to_name[self.args.dataset_id]} ({np.shape(x)}) loaded.')
        return x, y

    def generate_task(self, x, y):
        x_train, y_train = self.split_train_test_valid(x, y)
        s, u = self.split_s_and_u(x_train, y_train)
        self.split_s(s)
        self.split_u(u)

    def split_train_test_valid(self, x, y):
        self.num_examples = len(x)
        self.num_train = self.num_examples - (self.args.num_test+self.args.num_valid) 
        self.num_test = self.args.num_test
        self.labels = np.unique(y)
        # train set
        x_train = x[:self.num_train]
        y_train = y[:self.num_train]
        # test set
        x_test = x[self.num_train:self.num_train+self.num_test]
        y_test = y[self.num_train:self.num_train+self.num_test]
        y_test = tf.keras.utils.to_categorical(y_test, len(self.labels))
        l_test = np.unique(y_test)
        self.save_task({
            'x': x_test,
            'y': y_test,
            'labels': l_test,
            'name': f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        # valid set
        x_valid = x[self.num_train+self.num_test:]
        y_valid = y[self.num_train+self.num_test:]
        y_valid = tf.keras.utils.to_categorical(y_valid, len(self.labels))
        l_valid = np.unique(y_valid)
        self.save_task({
            'x': x_valid,
            'y': y_valid,
            'labels': l_valid,
            'name': f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}'
        })
        return x_train, y_train

    def split_s_and_u(self, x, y):
        if self.is_labels_at_server:
            self.num_s = self.args.num_labels_per_class
        else:
            self.num_s = self.args.num_labels_per_class * self.args.num_clients 

        data_by_label = {}
        for label in self.labels:
            idx = np.where(y[:]==label)[0] 
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        s_by_label, u_by_label = {}, {}
        for label, data in data_by_label.items():
            s_by_label[label] = {
                'x': data['x'][:self.num_s],
                'y': data['y'][:self.num_s]
            }
            u_by_label[label] = {
                'x': data['x'][self.num_s:],
                'y': data['y'][self.num_s:]
            }
            self.num_u += len(u_by_label[label]['x'])

        return s_by_label, u_by_label
        
    def split_s(self, s):
        if self.is_labels_at_server:
            x_labeled = []
            y_labeled = []
            for label, data in s.items():
                x_labeled = [*x_labeled, *data['x']]
                y_labeled = [*y_labeled, *data['y']]
            x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
            self.save_task({
                'x': x_labeled,
                'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}',
                'labels': np.unique(y_labeled)
            })
        else:
            for cid in range(self.args.num_clients):
                x_labeled = []
                y_labeled = []
                for label, data in s.items():
                    start = self.args.num_labels_per_class * cid
                    end = self.args.num_labels_per_class * (cid+1)
                    _x = data['x'][start:end]
                    _y = data['y'][start:end]
                    x_labeled = [*x_labeled, *_x]
                    y_labeled = [*y_labeled, *_y]
                x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
                self.save_task({
                    'x': x_labeled,
                    'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                    'name': f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_labeled)
                })

    def split_u(self, u):
        if self.is_imbalanced:
            ten_types_of_class_imbalanced_dist = [
                [0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15], # type 0
                [0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03], # type 1 
                [0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03], # type 2 
                [0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03], # type 3 
                [0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02], # type 4 
                [0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03], # type 5 
                [0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03], # type 6 
                [0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03], # type 7 
                [0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15], # type 8 
                [0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50], # type 9
            ]
            labels = list(u.keys())
            num_u_per_client = int(self.num_u/self.args.num_clients)
            offset_per_label = {label:0 for label in labels}
            for cid in range(self.args.num_clients):
                # batch-imbalanced
                x_unlabeled = []
                y_unlabeled = []
                dist_type = cid%len(labels)
                freqs = np.random.choice(labels, num_u_per_client, p=ten_types_of_class_imbalanced_dist[dist_type])
                frq = []
                for label, data in u.items():
                    num_instances = len(freqs[freqs==label])
                    frq.append(num_instances)
                    start = offset_per_label[label]
                    end = offset_per_label[label]+num_instances
                    x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                    y_unlabeled = [*y_unlabeled, *data['y'][start:end]] 
                    offset_per_label[label] = end
                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_unlabeled)
                })    
        else:
            # batch-iid
            for cid in range(self.args.num_clients):
                x_unlabeled = []
                y_unlabeled = []
                for label, data in u.items():
                    # print('>>> ', label, len(data['x']))
                    num_unlabels_per_class = int(len(data['x'])/self.args.num_clients)
                    start = num_unlabels_per_class * cid
                    end = num_unlabels_per_class * (cid+1)
                    x_unlabeled = [*x_unlabeled, *data['x'][start:end]]
                    y_unlabeled = [*y_unlabeled, *data['y'][start:end]]  
                x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)
                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{cid}',
                    'labels': np.unique(y_unlabeled)
                })

    def save_task(self, data):
        np_save(base_dir=self.base_dir, filename=f"{data['name']}.npy", data=data)
        print(f"filename:{data['name']}, labels:[{','.join(map(str, data['labels']))}], num_examples:{len(data['x'])}")
    
    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random.seed(self.args.seed)
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]











        
