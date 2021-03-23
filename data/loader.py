import os
import pdb
import glob
import numpy as np

from PIL import Image
from scipy.ndimage.interpolation import rotate, shift
from third_party.rand_augment.randaug import RandAugment

from misc.utils import *
from config import *

class DataLoader:

    def __init__(self, args):
        """ Data Loader

        Loads data corresponding to the current client
        Transforms and augments the given batch images if needed.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """

        self.args = args
        self.shape = (32,32,3)
        self.rand_augment = RandAugment()
        self.base_dir = os.path.join(self.args.dataset_path, self.args.task) 
        self.stats = [{
                'mean': [x/255 for x in [125.3,123.0,113.9]],
                'std': [x/255 for x in [63.0,62.1,66.7]]
            }, {
                'mean': [0.2190, 0.2190, 0.2190],
                'std': [0.3318, 0.3318, 0.3318]
            }]

    def get_s_by_id(self, client_id):
        task = np_load(self.base_dir, f's_{self.args.dataset_id_to_name[self.args.dataset_id]}_{client_id}.npy')
        return task['x'], task['y'], task['name']

    def get_u_by_id(self, client_id, task_id):
        path = os.path.join(self.base_dir, f'u_{self.args.dataset_id_to_name[self.args.dataset_id]}_{client_id}*')
        tasks = sorted([os.path.basename(p) for p in glob.glob(path)])
        task = np_load(self.base_dir, tasks[task_id])
        return task['x'], task['y'], task['name']

    def get_s_server(self):
        task = np_load(self.base_dir, f's_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        return task['x'], task['y'], task['name']

    def get_test(self):
        task = np_load(self.base_dir, f'test_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        return task['x'], task['y']

    def get_valid(self):
        task = np_load(self.base_dir, f'valid_{self.args.dataset_id_to_name[self.args.dataset_id]}.npy')
        return task['x'], task['y']

    def scale(self, x):
        x = x.astype(np.float32)/255
        return x

    def augment(self, images, soft=True):
        if soft:
            indices = np.arange(len(images)).tolist() 
            sampled = random.sample(indices, int(round(0.5*len(indices)))) # flip horizontally 50% 
            images[sampled] = np.fliplr(images[sampled])
            sampled = random.sample(sampled, int(round(0.25*len(sampled)))) # flip vertically 25% from above
            images[sampled] = np.flipud(images[sampled])
            return np.array([shift(img, [random.randint(-2, 2), random.randint(-2, 2), 0]) for img in images]) # random shift
        else:
            return np.array([np.array(self.rand_augment(Image.fromarray(np.reshape(img, self.shape)), M=random.randint(2,5))) for img in images])
