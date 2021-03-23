__author__ = "Wonyong Jeong"
__email__ = "wyjeong@kaist.ac.kr"

import gc
import cv2
import time
import random
import tensorflow as tf 

from PIL import Image
from tensorflow.keras import backend as K

from misc.utils import *
from modules.federated import ClientModule

class Client(ClientModule):

    def __init__(self, gid, args):
        """ FedMatch Client

        Performs fedmatch cleint algorithms 
        Inter-client consistency, agreement-based labeling, disjoint learning, etc.

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        super(Client, self).__init__(gid, args)
        self.kl_divergence = tf.keras.losses.KLDivergence()
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        self.init_model()

    def init_model(self):
        self.local_model = self.net.build_resnet9(decomposed=True)
        self.helpers = [self.net.build_resnet9(decomposed=False) for _ in range(self.args.num_helpers)]
        self.sig = self.net.get_sigma()
        self.psi = self.net.get_psi()
        for h in self.helpers:
            h.trainable = False

    def _init_state(self):
        self.train.set_details({
            'loss_fn_s': self.loss_fn_s,
            'loss_fn_u': self.loss_fn_u,
            'model': self.local_model,
            'trainables_s': self.sig,
            'trainables_u': self.psi,
            'batch_size': self.args.batch_size_client,
            'num_epochs': self.args.num_epochs_client,
        })

    def _train_one_round(self, client_id, curr_round, sigma, psi, helpers=None):
        self.train.cal_s2c(self.state['curr_round'], sigma, psi, helpers)
        self.set_weights(sigma, psi)
        if helpers == None:
            self.is_helper_available = False
        else:
            self.is_helper_available = True
            self.restore_helpers(helpers)
        self.train.train_one_round(self.state['curr_round'], self.state['round_cnt'], self.state['curr_task'])
        self.logger.save_current_state(self.state['client_id'], {
            's2c': self.train.get_s2c(),
            'c2s': self.train.get_c2s(),
            'scores': self.train.get_scores()
        })

    def loss_fn_s(self, x, y):
        # loss function for supervised learning
        x = self.loader.scale(x)
        y_pred = self.local_model(x)
        loss_s = self.cross_entropy(y, y_pred) * self.args.lambda_s
        return y_pred, loss_s

    def loss_fn_u(self, x):
        # loss function for unsupervised learning
        loss_u = 0
        y_pred = self.local_model(self.loader.scale(x))
        conf = np.where(np.max(y_pred.numpy(), axis=1)>=self.args.confidence)[0]
        if len(conf)>0:
            x_conf = self.loader.scale(x[conf])
            y_pred = K.gather(y_pred, conf)
            if True: # inter-client consistency
                if self.is_helper_available:
                    y_preds = [rm(x_conf).numpy() for rid, rm in enumerate(self.helpers)]
                    if self.state['curr_round']>0:
                        #inter-client consistency loss
                        for hid, pred in enumerate(y_preds): 
                            loss_u += (self.kl_divergence(pred, y_pred)/len(y_preds))*self.args.lambda_i
                else:
                    y_preds = None
                # Agreement-based Pseudo Labeling
                y_hard = self.local_model(self.loader.scale(self.loader.augment(x[conf], soft=False)))
                y_pseu = self.agreement_based_labeling(y_pred, y_preds)
                loss_u += self.cross_entropy(y_pseu, y_hard) * self.args.lambda_a
            else:
                y_hard = self.local_model(self.loader.scale(self.loader.augment(x[conf], soft=False)))
                loss_u += self.cross_entropy(y_pred, y_hard) * self.args.lambda_a
        # additional regularization
        for lid, psi in enumerate(self.psi): 
            # l1 regularization
            loss_u += tf.reduce_sum(tf.abs(psi)) * self.args.lambda_l1
            # l2 regularization
            loss_u += tf.math.reduce_sum(tf.math.square(self.sig[lid]-psi)) * self.args.lambda_l2
        return y_pred, loss_u, len(conf)

    def agreement_based_labeling(self, y_pred, y_preds=None):
        y_pseudo = np.array(y_pred)
        if self.is_helper_available:
            y_vote = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.args.num_classes)
            y_votes = np.sum([tf.keras.utils.to_categorical(np.argmax(y_rm, axis=1), self.args.num_classes) for y_rm in y_preds], axis=0)
            y_vote = np.sum([y_vote, y_votes], axis=0)
            y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_vote, axis=1), self.args.num_classes)
        else:
            y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.args.num_classes)
        return y_pseudo

    def restore_helpers(self, helper_weights):
        for hid, hwgts in enumerate(helper_weights):
            wgts = self.helpers[hid].get_weights()
            for i in range(len(wgts)):
                wgts[i] = self.sig[i].numpy() + hwgts[i] # sigma + psi
            self.helpers[hid].set_weights(wgts)

    def get_weights(self):
        if self.args.scenario == 'labels-at-client':
            sigs = [sig.numpy() for sig in self.sig]
            psis = [psi.numpy() for psi in self.psi] 
            return np.concatenate([sigs,psis], axis=0)
        elif self.args.scenario == 'labels-at-server':
            return [psi.numpy() for psi in self.psi]

    def set_weights(self, sigma, psi):
        for i, sig in enumerate(sigma):
            self.sig[i].assign(sig)
        for i, p in enumerate(psi):
            self.psi[i].assign(p)

    