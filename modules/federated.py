import os
import sys
import copy
import time
import random
import threading
import atexit
import tensorflow as tf
from datetime import datetime

from misc.utils import *
from misc.logger import Logger
from data.loader import DataLoader
from modules.nets import NetModule
from modules.train import TrainModule

class ServerModule:

    def __init__(self, args, Client):
        """ Superclass for Server Module

        This module contains common server functions, 
        such as laoding data, training global model, handling clients, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        self.args = args
        self.client = Client
        self.clients = {}
        self.threads = []
        self.updates = []
        self.task_names = []
        self.curr_round = -1
        self.limit_gpu_memory()
        self.logger = Logger(self.args) 
        self.loader = DataLoader(self.args)
        self.net = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger)
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        atexit.register(self.atexit)

    def limit_gpu_memory(self):
        """ Limiting gpu memories

        Tensorflow tends to occupy all gpu memory. Specify memory size if needed at config.py.
        Please set at least 6 or 7 memory for runing safely (w/o memory overflows). 
        """
        self.gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
        self.gpus = tf.config.list_physical_devices('GPU')
        if len(self.gpus)>0:
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu = self.gpus[gpu_id]
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu, 
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*self.args.gpu_mem)])

    def run(self):
        self.logger.print('server', 'server process has been started')
        self.load_data()
        self.build_network()
        self.net.init_state('server')
        self.net.set_init_params()
        self.train.init_state('server')
        self.train.set_details({
            'model': self.global_model,
            'loss_fn': self.loss_fn,
            'trainables': self.trainables,
            'num_epochs': self.args.num_epochs_server,
            'batch_size': self.args.batch_size_server,
        })
        self.create_clients()
        self.train_clients()

    def load_data(self):
        if self.args.scenario == 'labels-at-server':
            self.x_train, self.y_train, self.task_name = self.loader.get_s_server()
        else:
            self.x_train, self.y_train, self.task_name = None, None, None
        self.x_valid, self.y_valid =  self.loader.get_valid()
        self.x_test, self.y_test =  self.loader.get_test()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid) 
        self.train.set_task({
            'task_name':self.task_name,
            'x_train':self.x_train, 
            'y_train':self.y_train,
            'x_valid':self.x_valid, 
            'y_valid':self.y_valid, 
            'x_test':self.x_test, 
            'y_test':self.y_test, 
        })

    def create_clients(self):
        args_copied = copy.deepcopy(self.args)
        if len(tf.config.experimental.list_physical_devices('GPU'))>0:
            gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
            gpu_ids_real = [int(gid) for gid in self.args.gpu.split(',')]
            cid_offset = 0
            self.logger.print('server', 'creating client processes on gpus ... ')
            for i, gpu_id in enumerate(gpu_ids):
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    self.clients[gpu_id] = self.client(gpu_id, args_copied)
        else:
            self.logger.print('server', 'creating client processes on cpu ... ')
            num_parallel = 10
            self.clients = {i:self.client(i, args_copied) for i in range(num_parallel)}

    def train_clients(self):
        start_time = time.time()
        self.threads = []
        self.updates = []
        cids = np.arange(self.args.num_clients).tolist()
        num_connected = int(round(self.args.num_clients*self.args.frac_clients))
        for curr_round in range(self.args.num_rounds*self.args.num_tasks):
            self.curr_round = curr_round
            #####################################
            if self.args.scenario == 'labels-at-server':
                self.train_global_model()
            #####################################  
            self.connected_ids = np.random.choice(cids, num_connected, replace=False).tolist() # pick clients
            self.logger.print('server', f'training clients (round:{self.curr_round}, connected:{self.connected_ids})')
            self._train_clients()

        self.logger.print('server', 'all clients done')
        self.logger.print('server', 'server done. ({}s)'.format(time.time()-start_time))
        sys.exit()

    def aggregate(self, updates):
        if self.args.model == 'fedmatch':
            return self.train.uniform_average(updates)
        else:
            if self.args.fed_method == 'fedavg':
                return self.train.fedavg(updates)
            elif self.args.fed_method == 'fedprox':
                return self.train.fedprox(updates)

    def train_global_model(self):
        self.logger.print('server', 'training global_model')
        num_epochs = self.args.num_epochs_server_pretrain if self.curr_round == 0 else self.args.num_epochs_server
        self.train.train_global_model(self.curr_round, self.curr_round, num_epochs)

    def loss_fn(self, x, y):
        x = self.loader.scale(x)
        y_pred = self.global_model(x)
        loss = self.cross_entropy(y, y_pred) * self.args.lambda_s
        return y_pred, loss

    def atexit(self):
        for thrd in self.threads:
            thrd.join()
        self.logger.print('server', 'all client threads have been destroyed.' )


########################################################################################
########################################################################################
########################################################################################

class ClientModule:

    def __init__(self, gid, args):
        """ Superclass for Client Module 

        This module contains common client functions, 
        such as loading data, training local model, switching states, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        self.args = args
        self.state = {'gpu_id': gid}
        self.logger = Logger(self.args) 
        self.loader = DataLoader(self.args)
        self.net = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger)

    def train_one_round(self, client_id, curr_round, weights=None, sigma=None, psi=None, helpers=None):
        self.switch_state(client_id)
        if self.state['curr_task']<0:
            self.init_new_task()
        else:
            self.is_last_task = (self.state['curr_task']==self.args.num_tasks-1)
            self.is_last_round = (self.state['round_cnt']%self.args.num_rounds==0 and self.state['round_cnt']!=0)
            self.is_last = self.is_last_task and self.is_last_round
            if self.is_last_round or self.train.state['early_stop']:
                if self.is_last_task:
                    # if self.train.state['early_stop']:
                    #     self.train.evaluate()
                    self.stop()
                    return
                else:
                    self.init_new_task()
            else:
                self.load_data()
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        #######################################
        with tf.device('/device:GPU:{}'.format(self.state['gpu_id'])):
            if self.args.model == 'fedmatch':
                self._train_one_round(client_id, curr_round, sigma, psi, helpers)
            else:
                self._train_one_round(client_id, curr_round, weights)
        #######################################
        self.save_state()
        return (self.get_weights(), self.get_train_size(), self.state['client_id'], 
                                self.train.get_c2s(), self.train.get_s2c())
    
    def switch_state(self, client_id):
        if self.is_new(client_id):
            self.net.init_state(client_id)
            self.train.init_state(client_id)
            self.init_state(client_id)
        else: # load_state
            self.net.load_state(client_id)
            self.train.load_state(client_id)
            self.load_state(client_id)

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.check_pts, f'{client_id}_client.npy'))
    
    def init_state(self, client_id):
        self.state['client_id'] = client_id
        self.state['done'] = False
        self.state['curr_task'] =  -1
        self.state['task_names'] = []
        self._init_state()

    def load_state(self, client_id):
        self.state = np_load(self.args.check_pts, f'{client_id}_client.npy')

    def save_state(self):
        self.net.save_state()
        self.train.save_state()
        np_save(self.args.check_pts, f"{self.state['client_id']}_client.npy", self.state)

    def init_new_task(self):
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.load_data()

    def load_data(self):
        if self.args.scenario == 'labels-at-client':
            if 'simb' in self.args.task and self.state['curr_task']>0:
                self.x_unlabeled, self.y_unlabeled, task_name = \
                    self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
            else:
                self.x_labeled, self.y_labeled, task_name = \
                    self.loader.get_s_by_id(self.state['client_id'])
                self.x_unlabeled, self.y_unlabeled, task_name = \
                    self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
        elif self.args.scenario == 'labels-at-server':
            self.x_labeled, self.y_labeled = None, None
            self.x_unlabeled, self.y_unlabeled, task_name = \
                self.loader.get_u_by_id(self.state['client_id'], self.state['curr_task'])
        self.x_test, self.y_test =  self.loader.get_test()
        self.x_valid, self.y_valid =  self.loader.get_valid()
        self.x_test = self.loader.scale(self.x_test)
        self.x_valid = self.loader.scale(self.x_valid) 
        self.train.set_task({
            'task_name': task_name.replace('u_',''),
            'x_labeled':self.x_labeled, 
            'y_labeled':self.y_labeled,
            'x_unlabeled':self.x_unlabeled, 
            'y_unlabeled':self.y_unlabeled,
            'x_valid':self.x_valid, 
            'y_valid':self.y_valid, 
            'x_test':self.x_test, 
            'y_test':self.y_test, 
        })

    def get_train_size(self):
        train_size = len(self.x_unlabeled)
        if self.args.scenario == 'labels-at-client':
            train_size += len(self.x_labeled)
        return train_size

    def get_task_id(self):
        return self.state['curr_task']

    def get_client_id(self):
        return self.state['client_id']

    def stop(self):
        self.logger.print(self.state['client_id'], 'finished learning all tasks')
        self.logger.print(self.state['client_id'], 'done.')
        self.done = True
