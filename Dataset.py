from datetime import time
import os
import numpy as np
from numpy.lib.arraysetops import unique 
import pandas as pd
from pandas.core.frame import DataFrame
import copy
import datetime
import csv
import random
from collections import defaultdict
import sklearn
import sklearn.model_selection
from tqdm import tqdm

FILE = {'ml-1m':'ml1m-kg1m','ml-20m':'ml20m-kg500k','Book-Crossing':'bx-kg150k'}

class Dataset(object):
    def __init__ (self, num_test_negatives = 100, num_train_negatives = 4, dataset_name = 'ml-1m', kg_neighbor_size=16):
        self._data_path = f'data/ds/{dataset_name}'
        self._id_map_path = f'data/kg/{FILE[dataset_name]}/item_id2entity_id.txt'
        self._kg_path = f'data/kg/{FILE[dataset_name]}/kg.txt'
        self.dataset_name = dataset_name
        self.item_index_old2new = {}
        self.entity_id2index  = {}
        self.relation_id2index = {}
        self.all_item_index_old2new = {}
        self.kg = []
        self.kg_neighbor_size = kg_neighbor_size
        self.num_test_negatives = num_test_negatives
        self.num_train_negatives = num_train_negatives
        self.read_item_index_to_entity_id_file()
        self.num_users, self.num_items, self.num_items_in_kg, self.train_positves, self.train_negatives, self.test_positives, self.test_negatives, self.validation_data,\
            self.popular_items, self.unpopular_items, self.num_indexed_entity, self.num_indexed_relation, self.kg, self.adj_entity, self.adj_relation  = self.load_data(self._data_path)
        # self.target_items = random.sample(self.unpopular_items, 10)
        if os.path.exists(f'{self._data_path}/data_split.npz'):
            self.save_get_split(f'{self._data_path}/data_split.npz',save=False)
        else:
            self.X_train, self.Y_train, self.X_val, self.Y_val = self.get_train_instances(self.train_positves, self.train_negatives)
            self.X_test, self.Y_test = self.get_test_instances(self.test_positives, self.test_negatives)
            self.save_get_split(f'{self._data_path}/data_split.npz', X_train=self.X_train, Y_train = self.Y_train,  X_val = self.X_val, Y_val=self.Y_val,X_test=self.X_test, Y_test=self.Y_test, save=True)

    def save_get_split(self, split_path, X_train=None, Y_train=None, X_val=None, Y_val=None, X_test=None, Y_test=None, save=False):
        if save:
            user_input_train, item_input_train = X_train[0], X_train[1]
            user_input_val, item_input_val = X_val[0], X_val[1]
            user_input_test, item_input_test = X_test[0], X_test[1]
            np.savez(split_path,user_input_train=user_input_train, item_input_train=item_input_train, user_input_val = user_input_val, item_input_val=item_input_val,\
                 user_input_test= user_input_test,item_input_test=item_input_test,Y_train=Y_train,Y_val=Y_val,Y_test=Y_test )
        else:
            print('Load split ... ')
            data_split =  np.load(split_path)
            user_input_train,item_input_train, user_input_val, item_input_val, user_input_test, item_input_test, Y_train, Y_val, Y_test=\
                 data_split['user_input_train'],data_split['item_input_train'],data_split['user_input_val'],data_split['item_input_val'],\
                     data_split['user_input_test'],data_split['item_input_test'], data_split['Y_train'],data_split['Y_val'],data_split['Y_test']
            self.X_train = [user_input_train, item_input_train]
            self.Y_train = Y_train
            self.X_val = [user_input_val, item_input_val]
            self.Y_val = Y_val
            self.X_test = [user_input_test, item_input_test]
            self.Y_test = Y_test
                
    def get_all_data(self):
        return self.num_users, self.num_items, self.num_items_in_kg, self.test_positives, self.test_negatives, self.popular_items,\
            self.unpopular_items, self.num_indexed_entity, self.num_indexed_relation, self.kg, self.adj_entity, self.adj_relation
    
    def get_split(self):
        return self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test
    
    def read_item_index_to_entity_id_file(self):
        # file = './data/' + dataset + '/item_index2entity_id.txt'
        
        print('reading item index to entity id file: ' + self._id_map_path + ' ...')
        i = 0
        for line in open(self._id_map_path, encoding='utf-8').readlines():
            item_index = str(line.strip().split('\t')[0]) if self.dataset_name == 'Book-Crossing' else int(line.strip().split('\t')[0])
            satori_id = int(line.strip().split('\t')[1])
            self.item_index_old2new[item_index] = i
            self.entity_id2index[satori_id] = i
            i += 1
    
    def load_data(self, file_path):
        print(f'loading file {self.dataset_name} into dataframe...')
        if self.dataset_name == 'ml-1m':
            names = ['user_id', 'item_id', 'rating', 'timestamp']
            # user_id:int64 item_id:str rating:int64 timestamp:int64
            df = pd.read_csv(file_path+'/ratings.dat', index_col=False, sep='::', names=names, engine='python')
        elif self.dataset_name == 'ml-20m':
            names = ['user_id', 'item_id', 'rating', 'timestamp']
            # user_id:int64 item_id:str rating:float64 timestamp:int64
            df = pd.read_csv(file_path+'/ratings.csv', index_col=False, sep=',', names=names, engine='python', header=0)
        elif self.dataset_name == 'Book-Crossing':
            names = ['user_id', 'item_id', 'rating']
            #  encoding method is necessary in the process of reading dataset.
            #  user_id:int64 item_id:str rating:float64
            df = pd.read_csv(file_path+'/BX-Book-Ratings.csv',index_col=False, sep=';', names=names, header=0, encoding='cp1252')

            
        
        #reindex with base 0
        original_items = df['item_id'].unique()
        original_users = df['user_id'].unique()
        num_users = len(original_users)
        num_items = len(original_items) # bx 340557
        num_indexed_items = len(self.item_index_old2new)
        num_items_in_kg = len(self.item_index_old2new) # bx 14910
    
        for _, item in enumerate(original_items):
            if item not in self.item_index_old2new:
                self.item_index_old2new[item] = num_indexed_items
                num_indexed_items += 1
    
        user_map = {user: idx for idx, user in enumerate(original_users)}
        item_map = {item: self.item_index_old2new[item] for idx, item in enumerate(original_items)}
        print("Reindex dataframe...")
        df['item_id'] = df['item_id'].apply(lambda item:item_map[item])
        df['user_id'] = df['user_id'].apply(lambda user:user_map[user])
        rating_dict = {}
        item_count ,pop_ratio, unpop_ratio, unpopular_items , popular_items = [0 for i in range(num_items_in_kg)], 0.1, 0.9, [], [] 


        print("Store data into dictionary...")
        for row in df.itertuples():
            user_id = getattr(row, 'user_id')
            item_id = getattr(row, 'item_id')
            rating = getattr(row, 'rating')
            timestamp = 0 if self.dataset_name == 'Book-Crossing' else getattr(row, 'timestamp')
            if user_id not in rating_dict:
                rating_dict[user_id] = []
            rating_dict[user_id].append((item_id, rating, timestamp))
            # count the appearance
            if item_id < num_items_in_kg:
                item_count[item_id] += 1
        
        print("Get the popular and unpopular item (item in KG)")
        item_count_sort = sorted(item_count, reverse=True)
        popular_line = item_count_sort[int(len(np.nonzero(item_count_sort)[0])*pop_ratio)]
        unpopular_line = item_count_sort[int(len(np.nonzero(item_count_sort)[0])*unpop_ratio)]
        print(f'popular_line: {popular_line}, unpopular_line: {unpopular_line}')
        for i in range(num_items_in_kg):
            if item_count[i]<=unpopular_line:
                unpopular_items.append(i)
            if item_count[i]>=popular_line:
                popular_items.append(i)
        print(f'popular item num: {len(popular_items)}, unpopular item num: {len(unpopular_items)}')


        print('Load Knowledge Graph')
        kg = []
        num_indexed_entity = len(self.entity_id2index)
        num_indexed_relation = 0
        with open(self._kg_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                array = line.strip().split('\t')
                head_old = int(array[0])
                relation_old = array[1]
                tail_old = int(array[2])
                
                if head_old not in self.entity_id2index:
                    self.entity_id2index[head_old] = num_indexed_entity
                    num_indexed_entity += 1
                head = self.entity_id2index[head_old]

                if tail_old not in self.entity_id2index:
                    self.entity_id2index[tail_old] = num_indexed_entity
                    num_indexed_entity += 1
                tail = self.entity_id2index[tail_old]

                if relation_old not in self.relation_id2index:
                    self.relation_id2index[relation_old] = num_indexed_relation
                    num_indexed_relation += 1
                relation = self.relation_id2index[relation_old]
                kg.append((head, relation, tail))
        print('number of entities (containing items): %d' % num_indexed_entity)
        print('number of relations: %d' % num_indexed_relation)
        
        print('Construct_undirected_kg dict')
        kg_dict = defaultdict(list)
        for head_id, relation_id, tail_id in kg:
            kg_dict[head_id].append((relation_id, tail_id))
            kg_dict[tail_id].append((relation_id, head_id))

        print('Get KG adj List')
        adj_entity, adj_relation = [None for _ in range(num_indexed_entity)], [None for _ in range(num_indexed_entity)]
        for entity_id in range(num_indexed_entity):
            neighbors = kg_dict[entity_id]
            n_neighbor = len(neighbors)
            sample_indices = np.random.choice(range(n_neighbor), size=self.kg_neighbor_size, replace=n_neighbor < self.kg_neighbor_size)
            adj_relation[entity_id] = [neighbors[i][0] for i in sample_indices]
            adj_entity[entity_id] = [neighbors[i][1] for i in sample_indices]

        print('Load train test negative positive csv if exist')
        if os.path.exists(self._data_path + '/train_negatives.csv'):
            validation_data = []
            train_positives, train_negatives, test_positives, test_negatives = self.read_data_from_csv()
            return num_users, num_items, num_items_in_kg, train_positives, train_negatives, test_positives, test_negatives,validation_data,\
                 popular_items, unpopular_items, num_indexed_entity, num_indexed_relation, kg, adj_entity, adj_relation

        print("Sorting data according to timestamp...")
        for user_id in rating_dict:
            # print(f'before sort{rating_dict[user_id]}')
            rating_dict[user_id] = sorted(rating_dict[user_id],key=lambda x:(x[2]),reverse=True)
            # print(f'after sort{rating_dict[user_id]}')
            # exit()
        print("get train data and test data...")
        test_positives = []
        train_positives = []
        test_negatives = []
        train_negatives = []
        validation_data = []
        all_items = set(range(num_items))
        for user_id in tqdm(rating_dict.keys(),desc='get train&test'):
            rated_items = set([record[0] for record in rating_dict[user_id]])
            all_negatives = all_items.difference(rated_items)
            latest_record = rating_dict[user_id].pop(0)
            timestamp = latest_record[2]
            item_id = latest_record[0]
            rating = latest_record[1]
            test_positives.append((user_id, item_id, rating, timestamp))
            for record in rating_dict[user_id]:
                timestamp = record[2]
                item_id = record[0]
                rating = record[1]
                train_positives.append((user_id, item_id, rating, timestamp))
                sample_items = random.sample(all_negatives, self.num_train_negatives)
                train_negatives.append(sample_items)
            sample_items = random.sample(all_negatives, self.num_test_negatives)
            test_negatives.append(sample_items)
        assert len(train_positives) == len(train_negatives)
        print("writing data to csv...")
        # train_positives List(Tuple) 994169, train_negatives List(List) 994169 test_positives List(Tuple) 6040 , test_negatives List(List) 6040
        self.write_data_to_csv(train_positives, train_negatives, test_positives, test_negatives)

        return num_users, num_items, num_items_in_kg, train_positives, train_negatives, test_positives, test_negatives, validation_data,\
             popular_items, unpopular_items, num_indexed_entity, num_indexed_relation, kg, adj_entity, adj_relation

    def write_data_to_csv(self, train_positives, train_negatives, test_positives, test_negatives):
        with open(self._data_path + '/train_positives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for tup in train_positives:
                user_id = tup[0]
                item_id = tup[1]
                rating = tup[2]
                timestamp = tup[3]
                writer.writerow([user_id, item_id, rating, timestamp])
        with open(self._data_path + '/test_positives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for tup in test_positives:
                user_id = tup[0]
                item_id = tup[1]
                rating = tup[2]
                timestamp = tup[3]
                writer.writerow([user_id, item_id, rating, timestamp])

        with open(self._data_path + '/train_negatives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(train_negatives)):
                writer.writerow(train_negatives[i])
                
        with open(self._data_path + '/test_negatives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(test_negatives)):
                writer.writerow(test_negatives[i])
    
    def read_data_from_csv(self):
        test_positives = []
        train_positives = []
        test_negatives = []
        train_negatives = []
        with open(self._data_path + '/train_positives.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                user_id = int(row[0])
                item_id = int(row[1])
                rating = int(row[2]) if self.dataset_name == 'ml-1m' else float(row[2])
                timestamp = int(row[3])
                train_positives.append(tuple([user_id, item_id, rating, timestamp]))
        with open(self._data_path + '/test_positives.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                user_id = int(row[0])
                item_id = int(row[1])
                rating = int(row[2]) if self.dataset_name == 'ml-1m' else float(row[2])
                timestamp = int(row[3])
                test_positives.append(tuple([user_id, item_id, rating, timestamp]))

        with open(self._data_path + '/train_negatives.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                train_negatives.append([int(x) for x in row])
                
        with open(self._data_path + '/test_negatives.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                test_negatives.append([int(x) for x in row])
        return train_positives, train_negatives, test_positives, test_negatives

    def get_train_instances(self, train_positives, train_negatives):
        user_input, item_input, labels = [], [], []
        for i in range(len(train_positives)):
            record = train_positives[i]
            user = record[0]
            item = record[1]
            user_input.append(user)
            item_input.append(item)
            labels.append(1)
            for item in train_negatives[i]:
                user_input.append(user)
                item_input.append(item)
                labels.append(0)
        user_input = np.array(user_input)
        item_input = np.array(item_input)
        labels = np.array(labels)
        np.random.seed(200)
        np.random.shuffle(user_input)
        np.random.seed(200)
        np.random.shuffle(item_input)
        np.random.seed(200)
        np.random.shuffle(labels)
        user_input_train, user_input_val, labels_train, labels_val = sklearn.model_selection.train_test_split(user_input, labels, test_size=0.1, random_state=0)
        item_input_train, item_input_val, labels_train, labels_val = sklearn.model_selection.train_test_split(item_input, labels, test_size=0.1, random_state=0)
        X_train = [user_input_train, item_input_train]
        Y_train = labels_train
        X_val = [user_input_val, item_input_val]
        Y_val = labels_val 
        return X_train, Y_train, X_val, Y_val

    def get_test_instances(self, test_positives, test_negatives):
        user_input, item_input, labels = [], [], []
        for i in range(len(test_positives)):
            record = test_positives[i]
            user = record[0]
            item = record[1]
            user_input.append(user)
            item_input.append(item)
            labels.append(1)
            for item in test_negatives[i]:
                user_input.append(user)
                item_input.append(item)
                labels.append(0)
        np.random.seed(200)
        np.random.shuffle(user_input)
        np.random.seed(200)
        np.random.shuffle(item_input)
        np.random.seed(200)
        np.random.shuffle(labels)
        X_test = [np.array(user_input), np.array(item_input)]
        Y_test = np.array(labels)
        return X_test, Y_test
            
    def load_attack_data_get_attack_instances(self,attack_data):
        print('convert attack profiles to attack df')
        ui, user, item, r = {},[],[],[]
        for p in attack_data:
            u = p[0]
            for i in p[1:]:
                user.append(u)
                item.append(i)
                r.append(4)
        ui['user'] = user
        ui['item'] = item
        ui['rating'] = r
        attack_df = pd.DataFrame(ui)
        
        rating_dict = {}
        print("Store data into dictionary...")
        for row in attack_df.itertuples():
            user_id = getattr(row, 'user')
            item_id = getattr(row, 'item')
            rating = getattr(row, 'rating')
            if user_id not in rating_dict:
                rating_dict[user_id] = []
            rating_dict[user_id].append((item_id, rating))

        print('get attack fintune data')
        train_positives = []
        train_negatives = []
        all_items = set(range(self.num_items))
        for user_id in rating_dict:
            rated_items = set([record[0] for record in rating_dict[user_id]])
            all_negatives = all_items.difference(rated_items)
            for record in rating_dict[user_id]:
                item_id = record[0]
                rating = record[1]
                train_positives.append((user_id, item_id, rating))
                sample_items = random.sample(all_negatives, self.num_train_negatives)
                train_negatives.append(sample_items)
            assert len(train_positives) == len(train_negatives)
        print('get train instance')
        user_input, item_input, labels = [], [], []
        for i in range(len(train_positives)):
            record = train_positives[i]
            user = record[0]
            item = record[1]
            user_input.append(user)
            item_input.append(item)
            labels.append(1)
            for item in train_negatives[i]:
                user_input.append(user)
                item_input.append(item)
                labels.append(0)
        user_input = np.array(user_input)
        item_input = np.array(item_input)
        labels = np.array(labels)
        X_train = [user_input, item_input]
        Y_train = labels
        return X_train, Y_train
        
if __name__ == '__main__':
    dataset = Dataset()   
    print(dataset.num_items)