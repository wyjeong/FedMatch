import time
import math
import numpy as np
import tensorflow as tf 
import tensorflow.keras.metrics as tf_metrics

from misc.utils import *

class TrainModule:
    """ Common module for model training 

    This module manages training procedures for both server and client
    Saves and loads all states whenever client is switched.

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.metrics = {
            'train_lss': tf_metrics.Mean(name='train_lss'),
            'train_acc': tf_metrics.CategoricalAccuracy(name='train_acc'),
            'valid_lss': tf_metrics.Mean(name='valid_lss'),
            'valid_acc': tf_metrics.CategoricalAccuracy(name='valid_acc'),
            'test_lss' : tf_metrics.Mean(name='test_lss'),
            'test_acc' : tf_metrics.CategoricalAccuracy(name='test_acc')
        }

    def init_state(self, client_id):
        self.state = {
            'client_id': client_id,
            'scores': {
                'train_loss': [],
                'train_acc': [],
                'valid_loss': [],
                'valid_acc': [],
                'test_loss': [],
                'test_acc': [],
            },
            's2c': {
                'ratio': [],
                'sig_ratio': [],
                'psi_ratio': [],
                'hlp_ratio': [],
            },
            'c2s': {
                'ratio': [],
                'psi_ratio': [],
                'sig_ratio': [],
            },
            'total_num_params': 0,
            'optimizer_weights': []
        }
        self.init_learning_rate()

    def init_learning_rate(self):
        self.state['early_stop'] = False
        self.state['lowest_lss'] = np.inf
        self.state['curr_lr'] = self.args.lr
        self.state['curr_lr_patience'] = self.args.lr_patience
        self.init_optimizer(self.args.lr)

    def init_optimizer(self, curr_lr):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=curr_lr)

    def load_state(self, client_id):
        self.state = np_load(self.args.check_pts, f"{client_id}_train.npy")
        self.optimizer.set_weights(self.state['optimizer_weights'])

    def save_state(self):
        self.state['optimizer_weights'] = self.optimizer.get_weights()
        np_save(self.args.check_pts, f"{self.state['client_id']}_train.npy", self.state)

    def adaptive_lr_decay(self, vlss):
        if vlss<self.state['lowest_lss']:
            self.state['lowest_lss'] = vlss
            self.state['curr_lr_patience'] = self.args.lr_patience
        else:
            self.state['curr_lr_patience']-=1
            if self.state['curr_lr_patience']<=0:
                prev = self.state['curr_lr']
                self.state['curr_lr']/=self.args.lr_factor
                self.logger.print(self.state['client_id'], f"epoch:{self.state['curr_epoch']}, lr has been dropped")
                if self.state['curr_lr']<self.args.lr_min:
                    self.logger.print(self.state['client_id'], 'curr lr reached to the minimum')
                    # self.state['early_stop'] = True # not used to ensure synchronization
                self.state['curr_lr_patience'] = self.args.lr_patience
                self.optimizer.lr.assign(self.state['curr_lr'])

    def train_global_model(self, curr_round, round_cnt, num_epochs=None):
        num_epochs = self.params['num_epochs'] if num_epochs == None else num_epochs
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.num_train = len(self.task['x_train'])
        self.num_test = len(self.task['x_test'])
        start_time = time.time()            
        for epoch in range(num_epochs): 
            self.state['curr_epoch'] = epoch
            for i in range(0, len(self.task['x_train']), self.params['batch_size']): 
                x_batch = self.task['x_train'][i:i+self.params['batch_size']]
                y_batch = self.task['y_train'][i:i+self.params['batch_size']]
                with tf.GradientTape() as tape:
                    _, loss = self.params['loss_fn'](x_batch, y_batch)
                gradients = tape.gradient(loss, self.params['trainables']) 
                self.optimizer.apply_gradients(zip(gradients, self.params['trainables']))    
            vlss, vacc = self.validate()
            tlss, tacc = self.evaluate()
            self.adaptive_lr_decay(vlss)
            self.logger.print(self.state['client_id'], 'rnd:{}, ep:{}, n_train:{}, n_test:{} tlss:{}, tacc:{} ({}, {}s) '
                     .format(self.state['curr_round'], self.state['curr_epoch'], self.num_train, self.num_test, \
                            round(tlss, 4), round(tacc, 4), self.task['task_name'], round(time.time()-start_time,1)))
            if self.state['early_stop']:
                break

    def train_one_round(self, curr_round, round_cnt, curr_task):
        tf.keras.backend.set_learning_phase(1)
        
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.state['curr_task'] = curr_task
        if self.args.scenario == 'labels-at-client':
            bsize_s = self.params['batch_size']
            num_steps = round(len(self.task['x_labeled'])/bsize_s)
            bsize_u = math.ceil(len(self.task['x_unlabeled'])/num_steps)
        else:
            bsize_u = self.params['batch_size']
            num_steps = round(len(self.task['x_unlabeled'])/bsize_u)

        self.num_labeled = 0 if not isinstance(self.task['x_labeled'], np.ndarray) else len(self.task['x_labeled'])
        self.num_unlabeled = 0 if not isinstance(self.task['x_unlabeled'], np.ndarray) else len(self.task['x_unlabeled'])
        self.num_train = self.num_labeled + self.num_unlabeled
        self.num_test = len(self.task['x_test'])
        
        start_time = time.time()            
        for epoch in range(self.params['num_epochs']):
            self.state['curr_epoch'] = epoch
            self.num_confident = 0 
            for i in range(num_steps):
                if self.args.model in ['fedmatch']:
                    if self.args.scenario == 'labels-at-client':
                        ######################################
                        #         supervised learning    
                        ######################################
                        x_labeled = self.task['x_labeled'][i*bsize_s:(i+1)*bsize_s]
                        y_labeled = self.task['y_labeled'][i*bsize_s:(i+1)*bsize_s]
                        with tf.GradientTape() as tape:
                            _, loss_s = self.params['loss_fn_s'](x_labeled, y_labeled)
                        gradients = tape.gradient(loss_s, self.params['trainables_s']) 
                        self.optimizer.apply_gradients(zip(gradients, self.params['trainables_s'])) 
                    
                    x_unlabeled = self.task['x_unlabeled'][i*bsize_u:(i+1)*bsize_u] 
                    with tf.GradientTape() as tape:
                        ######################################
                        #       unsupervised learning    
                        ######################################
                        _, loss_u, num_conf = self.params['loss_fn_u'](x_unlabeled)
                    gradients = tape.gradient(loss_u, self.params['trainables_u']) 
                    self.optimizer.apply_gradients(zip(gradients, self.params['trainables_u'])) 
                    self.num_confident += num_conf
                else:
                    # base models: fixmatch & uda
                    x_unlabeled = self.task['x_unlabeled'][i*bsize_u:(i+1)*bsize_u] 
                    if len(x_unlabeled)>0:
                        with tf.GradientTape() as tape:
                            loss_final = 0
                            if self.args.scenario == 'labels-at-client':
                                x_labeled = self.task['x_labeled'][i*bsize_s:(i+1)*bsize_s]
                                y_labeled = self.task['y_labeled'][i*bsize_s:(i+1)*bsize_s]        
                                _, loss_s = self.params['loss_fn_s'](x_labeled, y_labeled)
                                loss_final += loss_s
                            _, loss_u, num_conf = self.params['loss_fn_u'](x_unlabeled)
                            loss_final += loss_u
                        if loss_final>0:
                            gradients = tape.gradient(loss_final, self.params['trainables']) 
                            self.optimizer.apply_gradients(zip(gradients, self.params['trainables'])) 
                        self.num_confident += num_conf

            vlss, vacc = self.validate()
            tlss, tacc = self.evaluate()
            if self.args.model in ['fedmatch']:
                ############################
                self.cal_c2s()
                ############################
                self.logger.print(self.state['client_id'], 
                    f"r:{self.state['curr_round']},"+
                    f"e:{self.state['curr_epoch']},"+
                    f"lss:{round(tlss, 4)},"+
                    f"acc:{round(tacc, 4)}, "+
                    f"n_train:{self.num_train}(s:{self.num_labeled},u:{self.num_unlabeled},c:{self.num_confident}), "+
                    f"n_test:{self.num_test}, "+
                    f"S2C:{round(self.s2c,2)}(s:{round(self.s2c_s,2)},p:{round(self.s2c_p,2)},h:{round(self.s2c_h,2)}), "+
                    f"C2S:{round(self.c2s,2)}(s:{round(self.c2s_s,2)},p:{round(self.c2s_p,2)} "+
                    f"({self.task['task_name']}, {round(time.time()-start_time,1)}s)")
                 
            else:
                self.logger.print(self.state['client_id'], 
                    f"rnd:{self.state['curr_round']},"+
                    f"ep:{self.state['curr_epoch']},"+
                    f"lss:{round(tlss, 4)},"+
                    f"acc:{round(tacc, 4)}, "+
                    f"n_train:{self.num_train}(s:{self.num_labeled},u:{self.num_unlabeled},c:{self.num_confident}), "+
                    f"n_test:{self.num_test}, "+
                    f"({self.task['task_name']}, {round(time.time()-start_time,1)}s)")

            self.adaptive_lr_decay(vlss)
            if self.state['early_stop']:
                break
        
    def validate(self):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_valid']), self.args.batch_size_test):
            x_batch = self.task['x_valid'][i:i+self.args.batch_size_test]
            y_batch = self.task['y_valid'][i:i+self.args.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('valid_lss', 'valid_acc', loss, y_batch, y_pred)
        vlss, vacc = self.measure_performance('valid_lss', 'valid_acc')
        self.state['scores']['valid_loss'].append(vlss)
        self.state['scores']['valid_acc'].append(vacc)
        return vlss, vacc
        
    def evaluate(self):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_test']), self.args.batch_size_test):
            x_batch = self.task['x_test'][i:i+self.args.batch_size_test]
            y_batch = self.task['y_test'][i:i+self.args.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
        tlss, tacc = self.measure_performance('test_lss', 'test_acc')
        self.state['scores']['test_loss'].append(tlss)
        self.state['scores']['test_acc'].append(tacc)
        return tlss, tacc

    def evaluate_forgetting(self):
        tf.keras.backend.set_learning_phase(0)
        x_labeled = self.scale(self.task['x_labeled'])
        for i in range(0, len(x_labeled), self.args.batch_size_test):
            x_batch = x_labeled[i:i+self.args.batch_size_test]
            y_batch = self.task['y_labeled'][i:i+self.args.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
        flss, facc = self.measure_performance('test_lss', 'test_acc')
        if not 'forgetting_acc' in self.state['scores']:
            self.state['scores']['forgetting_acc'] = []
        if not 'forgetting_loss' in self.state['scores']:
            self.state['scores']['forgetting_loss'] = []
        self.state['scores']['forgetting_loss'].append(flss)
        self.state['scores']['forgetting_acc'].append(facc)
        return flss, facc

    def evaluate_after_aggr(self):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_test']), self.args.batch_size_test):
            x_batch = self.task['x_test'][i:i+self.args.batch_size_test]
            y_batch = self.task['y_test'][i:i+self.args.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
        lss, acc = self.measure_performance('test_lss', 'test_acc')
        if not 'aggr_acc' in self.state['scores']:
            self.state['scores']['aggr_acc'] = []
        if not 'aggr_lss' in self.state['scores']:
            self.state['scores']['aggr_lss'] = []
        self.state['scores']['aggr_acc'].append(acc)
        self.state['scores']['aggr_lss'].append(lss)
        self.logger.print(self.state['client_id'], 'aggr_lss:{}, aggr_acc:{}'.format(round(lss, 4), round(acc, 4)))

    def add_performance(self, lss_name, acc_name, loss, y_true, y_pred):
        self.metrics[lss_name](loss)
        self.metrics[acc_name](y_true, y_pred)

    def measure_performance(self, lss_name, acc_name):
        lss = float(self.metrics[lss_name].result())
        acc = float(self.metrics[acc_name].result())
        self.metrics[lss_name].reset_states()
        self.metrics[acc_name].reset_states()
        return lss, acc

    def aggregate(self, updates):
        self.logger.print(self.state['client_id'], 'aggregating client-weights by {} ...'.format(self.args.fed_method))
        if self.args.fed_method == 'fedavg':
            return self.fedavg(updates)
        elif self.args.fed_method == 'fedprox':
            return self.fedprox(updates)
        else:
            print('no correct fedmethod was given: {}'.format(self.args.fed_method))
            os._exit(0)
    
    def fedavg(self, updates):
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(client_sizes[c]/total_size)
        return new_weights

    def fedprox(self, updates):
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(1/len(updates))
        return new_weights

    def uniform_average(self, updates):
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(1/len(updates))
        return new_weights

    def cal_s2c(self, curr_round, sig_server, psi_server, helpers):
        """Calculate S2C cost. 
        
        Nerual values that are meaningfully changed from server 
        will be updated at client in an element-wise manner.
        """
        if self.state['total_num_params'] == 0:
            for lid, psi in enumerate(self.params['trainables_u']):
                num_full = np.sum(np.ones(psi.shape))
                self.state['total_num_params'] += num_full
        
        self.set_server_weights(sig_server, psi_server)

        if curr_round == 0:
            sig_ratio = 1 
            psi_ratio = 1
            hlp_ratio = 0
            ratio = sig_ratio+psi_ratio+hlp_ratio
            self.state['s2c']['psi_ratio'].append(psi_ratio)
            self.state['s2c']['sig_ratio'].append(sig_ratio)
            self.state['s2c']['hlp_ratio'].append(hlp_ratio)
            self.state['s2c']['ratio'].append(ratio)
            self.s2c_s = sig_ratio
            self.s2c_p = psi_ratio
            self.s2c_h = hlp_ratio
            self.s2c = ratio
        else:
            sig_diff_list = []
            psi_diff_list = []
            total_psi_activs = 0
            total_sig_activs = 0
            for lid, psi_client in enumerate(self.params['trainables_u']):
                ##############################################
                # psi_server - psi_client 
                psi_server = self.sparsify(self.psi_server[lid])
                psi_client = self.sparsify(psi_client.numpy())
                psi_diff = self.cut(psi_server-psi_client)
                psi_activs = np.sum(np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32))
                total_psi_activs += psi_activs
                psi_diff_list.append(psi_diff)
                ##############################################
                # sig_server - sig_client 
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid].numpy()
                sig_diff = self.cut(sig_server - sig_client)
                sig_activs = np.sum(np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32))
                total_sig_activs += sig_activs
                sig_diff_list.append(sig_diff)
                ##############################################
            if helpers == None:
                hlp_ratio = 0
            else:
                total_hlp_activs = 0
                for hid, helper in enumerate(helpers):
                    for lid, hlp in enumerate(helper):
                        ##############################################
                        # hlp - psi_client 
                        hlp = self.sparsify(hlp)
                        hlp_mask = np.not_equal(hlp, np.zeros_like(hlp)).astype(np.float32)
                        psi_inter = hlp_mask * self.sparsify(self.params['trainables_u'][lid].numpy())
                        hlp_diff = self.cut(hlp-psi_inter)
                        ##############################################
                        hlp_activs = np.sum(np.not_equal(hlp_diff, np.zeros_like(hlp_diff)).astype(np.float32))
                        total_hlp_activs += hlp_activs
                hlp_ratio = total_hlp_activs/self.state['total_num_params']

            sig_ratio = total_sig_activs/self.state['total_num_params']           
            psi_ratio = total_psi_activs/self.state['total_num_params']
            ratio = psi_ratio + sig_ratio + hlp_ratio
            self.state['s2c']['sig_ratio'].append(sig_ratio)
            self.state['s2c']['psi_ratio'].append(psi_ratio)
            self.state['s2c']['hlp_ratio'].append(hlp_ratio)
            self.state['s2c']['ratio'].append(ratio)
            self.s2c_s = sig_ratio
            self.s2c_p = psi_ratio
            self.s2c_h = hlp_ratio
            self.s2c = ratio
            
            # update only changed elements, while fixing unchanged elements
            for lid in range(len(self.params['trainables_u'])):

                psi_diff = psi_diff_list[lid]
                psi_server = self.psi_server[lid]
                psi_client = self.params['trainables_u'][lid]
                psi_changed = psi_server*np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
                psi_unchanged = psi_client.numpy()*np.equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
                psi_client.assign(psi_changed+psi_unchanged)

                sig_diff = sig_diff_list[lid]
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid]
                sig_changed = sig_server*np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_unchanged = sig_client.numpy()*np.equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_client.assign(sig_changed+sig_unchanged)

    def cal_c2s(self):
        """Calculate C2S cost. 
        
        Nerual values that are meaningfully changed from client 
        will be updated at server in an element-wise manner.
        """
        if self.state['total_num_params'] == 0:
            for lid, psi in enumerate(self.params['trainables_u']):
                num_full = np.sum(np.ones(psi.shape))
                self.state['total_num_params'] += num_full
        sig_diff_list = []
        psi_diff_list = []
        total_psi_activs = 0
        total_sig_activs = 0
        for lid, psi_client in enumerate(self.params['trainables_u']):
            ##############################################
            # psi_client - psi_server
            psi_client = self.sparsify(psi_client.numpy())
            psi_server = self.psi_server[lid]
            psi_diff = self.cut(psi_client-psi_server)
            psi_activs = np.sum(np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32))
            total_psi_activs += psi_activs
            psi_diff_list.append(psi_diff)
            ##############################################
            # sig_client-sig_server
            if self.args.scenario == 'labels-at-client':
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid].numpy()
                sig_diff = self.cut(sig_client-sig_server)
                sig_activs = np.sum(np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32))
                total_sig_activs += sig_activs
                sig_diff_list.append(sig_diff)
            ##############################################
        
        psi_ratio = total_psi_activs/self.state['total_num_params']
        sig_ratio = total_sig_activs/self.state['total_num_params']
        ratio = psi_ratio + sig_ratio
        self.state['c2s']['sig_ratio'].append(sig_ratio)
        self.state['c2s']['psi_ratio'].append(psi_ratio)
        self.state['c2s']['ratio'].append(ratio)
        self.c2s_s = sig_ratio
        self.c2s_p = psi_ratio
        self.c2s = ratio
        
        # update only changed elements, while fixing unchanged elements
        for lid in range(len(self.params['trainables_u'])):
            psi_diff = psi_diff_list[lid]
            psi_server = self.psi_server[lid]
            psi_client = self.params['trainables_u'][lid]
            psi_changed = psi_client.numpy()*np.not_equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
            psi_unchanged = psi_server*np.equal(psi_diff, np.zeros_like(psi_diff)).astype(np.float32)
            psi_client.assign(psi_changed+psi_unchanged)

            if self.args.scenario == 'labels-at-client':
                sig_diff = sig_diff_list[lid]
                sig_server = self.sig_server[lid]
                sig_client = self.params['trainables_s'][lid]
                sig_changed = sig_client.numpy()*np.not_equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_unchanged = sig_server*np.equal(sig_diff, np.zeros_like(sig_diff)).astype(np.float32)
                sig_client.assign(sig_changed+sig_unchanged)


    def scale(self, images):
        return images.astype(np.float32)/255.

    def sparsify(self, weights):
        hard_threshold = tf.cast(tf.greater(tf.abs(weights), self.args.l1_thres), tf.float32)
        return tf.multiply(weights, hard_threshold)

    def cut(self, weights):
        hard_threshold = tf.cast(tf.greater(tf.abs(weights), self.args.delta_thres), tf.float32)
        return tf.multiply(weights, hard_threshold)

    def set_server_weights(self, sig_server, psi_server):
        self.sig_server = sig_server 
        self.psi_server = psi_server 

    def get_s2c(self):
        return self.state['s2c']

    def get_c2s(self):
        return self.state['c2s']

    def set_details(self, details):
        self.params = details

    def set_task(self, task):
        self.task = task

    def get_scores(self):
        return self.state['scores']
    
    def get_train_size(self):
        train_size = len(self.task['x_unlabeled'])
        if self.args.scenario == 'labels-at-client':
            train_size += len(self.task['x_labeled'])
        return train_size

    
