import threading
import tensorflow as tf 

from scipy import spatial
from scipy.stats import truncnorm

from misc.utils import *
from models.fedmatch.client import Client
from modules.federated import ServerModule

class Server(ServerModule):

    def __init__(self, args):
        """ FedMatch Server

        Performs fedmatch server algorithms 
        Embeds local models and searches nearest if requird

        Created by:
            Wonyong Jeong (wyjeong@kaist.ac.kr)
        """
        super(Server, self).__init__(args, Client)
        self.c2s_sum = []
        self.c2s_sig = []
        self.c2s_psi = []
        self.s2c_sum = []
        self.s2c_sig = []
        self.s2c_psi = []
        self.s2c_hlp = []
        self.restored_clients = {}
        self.rid_to_cid = {}
        self.cid_to_vectors = {}
        self.cid_to_weights = {}
        self.curr_round = -1
        mu,std,lower,upper = 125,125,0,255
        self.rgauss = self.loader.scale(truncnorm((lower-mu)/std,(upper-mu)/std, 
                        loc=mu, scale=std).rvs((1,32,32,3))) # fixed gaussian noise for model embedding

    def build_network(self):
        self.global_model = self.net.build_resnet9(decomposed=True)
        self.sig = self.net.get_sigma()
        self.psi = self.net.get_psi()
        self.trainables = [sig for sig in self.sig] # only sigma will be updated at server (Labels at Serve scenario)
        num_connected = int(round(self.args.num_clients*self.args.frac_clients))
        self.restored_clients = {i:self.net.build_resnet9(decomposed=False) for i in range(num_connected)}
        for rid, rm in self.restored_clients.items():
            rm.trainable = False

    def _train_clients(self):
        sigma = [s.numpy() for s in self.sig]
        psi = [p.numpy() for p in self.psi]
        while len(self.connected_ids)>0:
            for gpu_id, gpu_client in self.clients.items():
                cid = self.connected_ids.pop(0)
                helpers = self.get_similar_models(cid)
                with tf.device('/device:GPU:{}'.format(gpu_id)): 
                    # each client will be trained in parallel 
                    thrd = threading.Thread(target=self.invoke_client, args=(gpu_client, cid, self.curr_round, sigma, psi, helpers))
                    self.threads.append(thrd)
                    thrd.start()
                if len(self.connected_ids) == 0:
                    break
            # wait all threads per gpu
            for thrd in self.threads:
                thrd.join()   
            self.threads = []

        self.client_similarity(self.updates)
        self.set_weights(self.aggregate(self.updates))
        self.train.evaluate_after_aggr()
        self.avg_c2s()
        self.avg_s2c()
        self.logger.save_current_state('server', {
            'c2s': {
                'sum': self.c2s_sum,
                'sig': self.c2s_sig,
                'psi': self.c2s_psi,
            },
            's2c': {
                'sum': self.s2c_sum,
                'sig': self.s2c_sig,
                'psi': self.s2c_psi,
                'hlp': self.s2c_hlp,
            },
            'scores': self.train.get_scores()
        }) 
        self.updates = []

    def invoke_client(self, client, cid, curr_round, sigma, psi, helpers):
        update = client.train_one_round(cid, curr_round, sigma=sigma, psi=psi, helpers=helpers)
        self.updates.append(update)

    def client_similarity(self, updates):
        self.restore_clients(updates)
        for rid, rmodel in self.restored_clients.items():
            cid = self.rid_to_cid[rid]
            self.cid_to_vectors[cid] = np.squeeze(rmodel(self.rgauss)) # embed models
        self.vid_to_cid = list(self.cid_to_vectors.keys())
        self.vectors = list(self.cid_to_vectors.values())
        self.tree = spatial.KDTree(self.vectors)
    
    def restore_clients(self, updates):
        rid = 0
        self.rid_to_cid = {}
        for cwgts, csize, cid, _, _ in updates:
            self.cid_to_weights[cid] = cwgts
            rwgts = self.restored_clients[rid].get_weights()
            if self.args.scenario == 'labels-at-client':
                half = len(cwgts)//2
                for lid in range(len(rwgts)):
                    rwgts[lid] = cwgts[lid] + cwgts[lid+half] # sigma + psi
            elif self.args.scenario == 'labels-at-server':
                for lid in range(len(rwgts)):
                    rwgts[lid] = self.sig[lid].numpy() + cwgts[lid] # sigma + psi
            self.restored_clients[rid].set_weights(rwgts)
            self.rid_to_cid[rid] = cid
            rid += 1

    def get_similar_models(self, cid):
        if cid in self.cid_to_vectors and (self.curr_round+1)%self.args.h_interval == 0:
            cout = self.cid_to_vectors[cid]
            sims = self.tree.query(cout, self.args.num_helpers+1)
            hids = []
            weights = []
            for vid in sims[1]:
                selected_cid = self.vid_to_cid[vid]
                if selected_cid == cid:
                    continue
                w = self.cid_to_weights[selected_cid]
                if self.args.scenario == 'labels-at-client':
                    half = len(w)//2
                    w = w[half:]
                weights.append(w)
                hids.append(selected_cid)
            return weights[:self.args.num_helpers]
        else:
            return None 

    def set_weights(self, new_weights):
        if self.args.scenario == 'labels-at-client':
            half = len(new_weights)//2
            for i, nwghts in enumerate(new_weights):
                if i < half:
                    self.sig[i].assign(new_weights[i])
                else:
                    self.psi[i-half].assign(new_weights[i])
        elif self.args.scenario == 'labels-at-server':
            for i, nwghts in enumerate(new_weights):
                self.psi[i].assign(new_weights[i])
    
    def avg_c2s(self): # client-wise average
        ratio_list = []
        sig_list = []
        psi_list = []
        for upd in self.updates:
            c2s = upd[3]
            ratio_list.append(c2s['ratio'][-1])
            sig_list.append(c2s['sig_ratio'][-1])
            psi_list.append(c2s['psi_ratio'][-1])
        try:
            self.c2s_sum.append(np.mean(ratio_list, axis=0))
            self.c2s_sig.append(np.mean(sig_list, axis=0))
            self.c2s_psi.append(np.mean(psi_list, axis=0))
        except:
            pdb.set_trace()

    def avg_s2c(self): # client-wise average
        sum_list = []
        sig_list = []
        psi_list = []
        hlp_list = []
        for upd in self.updates:
            s2c = upd[4]
            sum_list.append(s2c['ratio'][-1])
            sig_list.append(s2c['sig_ratio'][-1])
            psi_list.append(s2c['psi_ratio'][-1])
            hlp_list.append(s2c['hlp_ratio'][-1])
        self.s2c_sum.append(np.mean(sum_list, axis=0))
        self.s2c_sig.append(np.mean(sig_list, axis=0))
        self.s2c_psi.append(np.mean(psi_list, axis=0))
        self.s2c_hlp.append(np.mean(hlp_list, axis=0))
    
    
