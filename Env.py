from tqdm import tqdm
import numpy as np
import copy
import time
import random
random.seed(10)
import os
import json
import pickle
import pandas as pd
from NeuMF import NeuMF
from Dataset import Dataset
import tensorflow as tf
import torch

class Env():
    def __init__(self,eval_candi_num, max_candi_num, attack_topk, attack_epoch, target_item_num, n_max_attacker, episode_length, eval_spy_users_num, train_spy_users_num, \
        dataset_name='ml1m', processed_path = 'processed_data'):
        
        self.dataset_name = dataset_name
        self.processed_path = processed_path
        self.train_spy_users_num = train_spy_users_num
        self.eval_spy_users_num = eval_spy_users_num
        self.episode_length = episode_length
        self.n_max_attacker=n_max_attacker
        self.target_item_num = target_item_num
        self.attack_epoch = attack_epoch
        self.attack_topk = attack_topk
        self.max_candi_num = max_candi_num
        self.eval_candi_num = eval_candi_num
        self._init = 1
        assert ( self.eval_candi_num <= self.max_candi_num )

        print('Init dataset')

        self.dataset = Dataset(num_test_negatives = max_candi_num, num_train_negatives = 4, dataset_name = self.dataset_name, kg_neighbor_size=16)

        self.num_users, self.num_items, self.num_items_in_kg, self.test_positives, self.test_negatives,\
                self.popular_items, self.unpopular_items, self.num_indexed_entity, self.num_indexed_relation,\
                     self.kg, self.adj_entity, self.adj_relation = self.dataset.get_all_data()
        
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = self.dataset.get_split()

        self.target_items = random.sample(self.unpopular_items, self.target_item_num)
        print(f'target items = {self.target_items}')

        self.adj_item_dict = self.find_neighbor_items(self.adj_entity, self.num_items_in_kg)
        
        ''' Rec Model'''
        self.model_n_users = self.num_users + self.n_max_attacker
        self.model_n_items = self.num_items
        if self.dataset_name == 'Book-Crossing' or self.dataset_name == 'ml-20m':
            self.model_layers = [8] # dropout + normal [16,8] k=3 / loss: 0.3593 - acc: 0.8563 - val_loss: 0.4008 - val_acc: 0.8511 // dropout0.4 + normal [32,16,8] k=8 loss: 0.3746 - acc: 0.8517 - val_loss: 0.3912 - val_acc: 0.8530
            self.reg_mf = 0 # dropout + normal 5e-3 5e-3 [16,8] k=5 / epoch 3 loss: 0.3647 - acc: 0.8542 - val_loss: 0.4055 - val_acc: 0.8504
            # dropout + normal 5e-2 5e-2 [16,8] k=5 / loss: 0.3473 - acc: 0.8584 - val_loss: 0.4111 - val_acc: 0.8509
            # dropout + normal 5e-1 5e-1 [16,8] k=5 / loss: 0.3561 - acc: 0.8572 - val_loss: 0.4057 - val_acc: 0.8509 ** hr = 0.600, ndcg = 0.311
            # 5e-1 8 k=5 dropout+normal还可以
            self.reg_mlp = [5e-1] #  dropout0.2 + normal [32,16,8] k=8 : loss: 0.3821 - acc: 0.8509 - val_loss: 0.3917 -val_acc: 0.8531
            self.k = 5 # 
            self.train_epoch = 30
        else:
            self.model_layers = [32, 16, 8]
            self.reg_mf = 0
            self.reg_mlp = [0, 0, 0, 0]
            self.k = 8
            self.train_epoch = 30
        self.learning_rate = 0.0005
        self.train_batch_size = 256

        self.rs, self.model = self.build_model(
                self.model_n_users,
                self.model_n_items, 
                self.model_layers,
                self.reg_mf,
                self.reg_mlp, 
                self.k, 
                self.learning_rate, 
                self.train_epoch, 
                self.train_batch_size)
    
    def get_env_meta(self):
        return self.num_users, self.num_items, self.num_items_in_kg, self.target_items, self.kg, self.num_indexed_entity, self.num_indexed_relation, \
            self.adj_entity, self.adj_relation, self.adj_item_dict
    
    def build_model(self, num_users, num_items, model_layers, reg_mf, reg_mlp, k, learning_rate, train_epoch, batch_size):
        print('build model...')
        if os.path.exists(f'{self.processed_path}/rec_model_epoch_{self.train_epoch}_max_user_{self.n_max_attacker}'):
            ncf = NeuMF(num_users, num_items, model_layers, reg_mf, reg_mlp, k)
            # model = ncf.get_model()
            # model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])
            # model.load_weights(f'{self.processed_path}/rec_model_epoch_{self.train_epoch}_max_user_{self.n_max_attacker}.hdf5')
            model = tf.keras.models.load_model(f'{self.processed_path}/rec_model_epoch_{self.train_epoch}_max_user_{self.n_max_attacker}')
            return ncf, model
        else:
            
            ncf = NeuMF(num_users, num_items, model_layers, reg_mf, reg_mlp, k)
            model = ncf.get_model()
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])
            print(model.summary())
            
            '''
            hits, ndcgs = ncf.evalueate_model(model, test_data, test_negatives)
            hr = np.array(hits).mean()
            ndcg = np.array(ndcgs).mean()
            '''

            print('start training...')
            best_hr = 0
            best_ndcg= 0
            checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{self.processed_path}/rec_model_epoch_{self.train_epoch}_max_user_{self.n_max_attacker}', monitor='loss', verbose=1,
                save_best_only=True, save_weights_only = False, mode='auto', period=1)
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=3)
            history = model.fit(
                self.X_train,
                self.Y_train,
                batch_size=batch_size,
                epochs=train_epoch,
                validation_data=(self.X_val, self.Y_val),
                shuffle=True,
                callbacks=[checkpoint, earlystopping]
            )
            # model.save_weights(f'{self.processed_path}/rec_model_epoch_{self.train_epoch}_max_user_{self.n_max_attacker}.h5')
            # model.save(f'{self.processed_path}/rec_model_epoch_{self.train_epoch}_max_user_{self.n_max_attacker}')
            return ncf, model
    
    def attack(self, attack_data, target_item, eval=False):
        print('evaluate attack performance: HR & NDCG')
        if not os.path.exists(f'{self.processed_path}/train_spy_idx.npy'):
            train_spy_idx = random.sample(list(range(len(self.test_positives))), self.train_spy_users_num)
            residual = list(set( list(range(len(self.test_positives))) ) - set(train_spy_idx))
            eval_spy_idx = random.sample(residual, self.eval_spy_users_num)
            np.save(f'{self.processed_path}/train_spy_idx.npy',train_spy_idx)
            np.save(f'{self.processed_path}/eval_spy_idx.npy',eval_spy_idx)
        else:
            train_spy_idx = np.load(f'{self.processed_path}/train_spy_idx.npy')
            eval_spy_idx = np.load(f'{self.processed_path}/eval_spy_idx.npy')
        spy_test_data = []
        spy_test_negatives = []
        if eval:
            idx = eval_spy_idx
        else:
            idx = train_spy_idx
        for i in idx:
            spy_test_data.append(self.test_positives[i][:self.eval_candi_num])
            spy_test_negatives.append(self.test_negatives[i][:self.eval_candi_num])
        # log train performance for experiments
        eval_spy_test_data = []
        eval_spy_test_negatives = []
        for i in eval_spy_idx:
            eval_spy_test_data.append(self.test_positives[i][:self.eval_candi_num])
            eval_spy_test_negatives.append(self.test_negatives[i][:self.eval_candi_num])
        
        X_train_attack, Y_train_attack = self.dataset.load_attack_data_get_attack_instances(attack_data)

        if eval:
            print('EVAL: Reset rec model')
            self.eval_rs, self.eval_model = self.build_model(
                self.model_n_users, 
                self.model_n_items, 
                self.model_layers, 
                self.reg_mf, 
                self.reg_mlp, 
                self.k, 
                self.learning_rate, 
                self.train_epoch, 
                self.train_batch_size)
            if self._init == 1:
                self._init = 0
                hits, ndcgs = self.eval_rs.evaluate_model(self.eval_model, spy_test_data, spy_test_negatives, target_item, self.attack_topk)
                with open(f'{self.processed_path}/before_attack_result.txt','a+') as f:
                    f.write(f'target_item: {target_item}, evaluate result: hit {np.array(hits).mean()}, ndcg {np.array(ndcgs).mean()} topk{self.attack_topk}')
                    f.write('\n')
            
            
            history = self.eval_model.fit(
                X_train_attack,
                Y_train_attack,
                batch_size=self.train_batch_size,
                epochs=self.attack_epoch,
                validation_data=(self.X_val, self.Y_val),
                shuffle=True
            )
        else:
            history = self.model.fit(
                X_train_attack,
                Y_train_attack,
                batch_size=self.train_batch_size,
                epochs=self.attack_epoch,
                validation_data=(self.X_val, self.Y_val),
                shuffle=True
            )

        
        
        if eval:
            hits, ndcgs = self.eval_rs.evaluate_model(self.eval_model, spy_test_data, spy_test_negatives, target_item, self.attack_topk)
            eval_hits, eval_ndcgs = hits, ndcgs
        else:
            hits, ndcgs = self.rs.evaluate_model(self.model, spy_test_data, spy_test_negatives, target_item, self.attack_topk)
            # log train performance
            eval_hits, eval_ndcgs = self.rs.evaluate_model(self.model, eval_spy_test_data, eval_spy_test_negatives, target_item, self.attack_topk)

        hr = np.array(hits).mean()
        ndcg = np.array(ndcgs).mean()
        print('attack performance: hr = %.3f, ndcg = %.3f' % (hr, ndcg))
        eval_hr = np.array(eval_hits).mean()
        eval_ndcg = np.array(eval_ndcgs).mean()
        print('attack performance on eval spy user: hr = %.3f, ndcg = %.3f' % (eval_hr, eval_ndcg))
        return hr, ndcg, eval_hr, eval_ndcg
      

    def set_target_items(self, target_items):
        print(f'Set target items: {target_items}')
        self.target_items = copy.deepcopy(target_items)
    
    def find_neighbor_items(self, adj_entity, item_count):
        if os.path.exists(f'{self.processed_path}/adj_item.pkl'):
            print('adj_item file exist')
            with open(f'{self.processed_path}/adj_item.pkl','rb') as fin:
                adj_item = pickle.load(fin)
                # adj_item = json.load(fin)
            #candi_dict = utils.pickle_load('./neighbors.pkl')
            #print(type(adj_item))
            a = 999
            cnt = 0
            for i in range(item_count):
                if a > len(adj_item[i]):
                    a = len(adj_item[i])
                if len(adj_item[i]) == 0:
                    cnt+=1
            print(f'have checked, adj_item min len = {a}, zero cnt = {cnt}, now return adj_item') # ml-20m: adj_item min len = 1, zero cnt = 0
            return adj_item
        else:
            # 给定kg 三元组，给entity(item)找到他们的对应的neighbor entity(item):
            hop = 5
            adj_item = {}
            item_set = set([i for i in range(item_count)])
            for item_new_id in range(item_count):
                adj_item[item_new_id] = set()
            for item_new_id in range(item_count):
                seed = adj_entity[item_new_id]
                for k in range(hop):
                    tmp = copy.deepcopy(seed)
                    for item in tmp:
                        seed.extend(adj_entity[item])
                    seed = list(set(seed))
                seedset = set(seed)
                #print(f'{item_new_id}before interact len{len(seedset)}')
                seedset = seedset & item_set 
                #print(f'{item_new_id}after interact len{len(seedset)}')
                adj_item[item_new_id]=list(seedset)
            a = 999
            cnt = 0
            for i in range(item_count):
                if a > len(adj_item[i]):
                    a = len(adj_item[i])
                if len(adj_item[i]) == 0:
                    cnt+=1
            print(f'have checked, adj_item min len = {a}, zero cnt = {cnt}, now save adj_item')
            with open(f'{self.processed_path}/adj_item.pkl','wb') as fin:
                pickle.dump(adj_item, fin)
            return adj_item

    def attack_reset(self, user_id):
        self.attackers_id = user_id
        self.step_count = 0
        self.history_items = set()
        self.state = np.array(self.attackers_id)[:, None]
        # self.state = np.tile(self.state, 1)[:,None]
        state = copy.deepcopy(self.state)
        
        return state

    def attack_step(self, action_chosen, target_item, evaluate = False, metric = 'hr'):
        '''
        input: 
            action_chosen: [B] 每一行都有一个新的action(item id)
        return 
            reward: [bs*attack_user_num] 到了episode长度时，attack_user_num个profile共同得到一个reward，一共有bs个reward，并需要normalize处理，然后repeat
            done: 到了episode长度时，返回True，否则返回False
        '''
        action_chosen = np.expand_dims(action_chosen, 1)
        
        # 生成new_state
        self.state = np.hstack((self.state, action_chosen))
        new_state = copy.deepcopy(self.state)

        # 生成reward 和 done
        reward = 0
        done = False
        self.step_count += 1
        
        # test = self.n_attacker
        if self.step_count == self.episode_length:
            rewards = []
            attack_set = []
            for i in range(len(self.attackers_id)):
                # u_id = self.state[i][0]
                user_profile = self.state[i]
                attack_set.append(user_profile)
                
            # attack_dataset 传入到RecSys中
            # print('attack_set = ',attack_set)
            hr, ndcg, eval_hr, eval_ndcg = self.attack(attack_set, target_item, evaluate)
            
            if metric == 'hr':
                rewards.append(hr)
            elif metric == 'ndcg':
                rewards.append(ndcg)


            user_rewards = []
            for t in rewards:
                user_rewards.extend([t for j in range(len(self.attackers_id))])
            user_rewards = np.array(user_rewards)
            
            reward = user_rewards
            done = True


        else:
            hr = -1
            ndcg = -1
            eval_hr = -1
            eval_ndcg = -1
            reward = np.zeros(len(self.attackers_id))
            done = False
        
        
        return new_state, reward, done, hr, ndcg, eval_hr, eval_ndcg

