import random
import tensorflow as tf 
import numpy as np
from Dataset import Dataset
import heapq
import math
import os
import copy
import matplotlib.pyplot as plt 
import sklearn.model_selection

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

class NeuMF(object):
    def __init__(self, num_users, num_items,model_layers, reg_mf, reg_mlp, k):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k 
        self.model_layers = model_layers
        self.reg_mf = reg_mf
        self.reg_mlp = reg_mlp
    
    def get_model(self):
        #define input is a 1-d array
        
        user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='user_input')
        item_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='item_input')

        #defile the way to initialize the keras layer
        embedding_init = tf.keras.initializers.RandomNormal(stddev=0.01)
        mf_embedding_user = tf.keras.layers.Embedding(
            self.num_users, 
            self.k,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mf),
            input_length=1,
            name='mf_embedding_user'
            )
        mf_embedding_item = tf.keras.layers.Embedding(
            self.num_items, 
            self.k,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mf),
            input_length=1,
            name='mf_embedding_item'
            )
        mlp_embedding_user = tf.keras.layers.Embedding(
            self.num_users, 
            self.model_layers[0]//2,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mlp[0]),
            input_length=1,
            name='mlp_embedding_user'
            )
        mlp_embedding_item = tf.keras.layers.Embedding(
            self.num_items, 
            self.model_layers[0]//2,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mlp[0]),
            input_length=1,
            name='mlp_embedding_item'
            )

        #GMF latent
        mf_user_latent = tf.keras.layers.Flatten()(mf_embedding_user(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(mf_embedding_item(item_input))
        gmf_layer = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

        #MLP latent
        mlp_user_latent = tf.keras.layers.Flatten()(mlp_embedding_user(user_input))
        mlp_item_latent = tf.keras.layers.Flatten()(mlp_embedding_item(item_input))
        mlp_layer = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

        num_layers = len(self.model_layers)
        for i in range(1, num_layers):
            new_layer = tf.keras.layers.Dense(
                self.model_layers[i],
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_mlp[i]),
                activation='relu',
                name = 'layer%d' %i
            )
            mlp_layer = new_layer(mlp_layer)
            new_layer = tf.keras.layers.Dropout(0.5)
            mlp_layer = new_layer(mlp_layer)    
            new_layer = tf.keras.layers.BatchNormalization()
            mlp_layer = new_layer(mlp_layer)
        
        NeuMF_layer = tf.keras.layers.concatenate([gmf_layer, mlp_layer])
        prediction = tf.keras.layers.Dense(
            1,
            kernel_initializer='lecun_uniform',
            activation='sigmoid',
            name='prediction'
        )(NeuMF_layer)

        model = tf.keras.models.Model(inputs=[user_input, item_input], outputs = prediction)
        return model

    def init_from_pretrain_model(self, model, gmf_model, mlp_model, model_layers):
        pass 

    def evaluate_model(self, model, test_data, test_negatives, target_item = None, attack_topk = 20):
        hits = []
        ndcgs = []
        for i in range(len(test_data)):
            hr, ndcg = self.evaluate_one_sample(model, i, test_data, test_negatives, target_item, attack_topk)
            hits.append(hr)
            ndcgs.append(ndcg)
        return hits, ndcgs 

    def evaluate_one_sample(self, model, idx, test_data, test_negatives, target_item, attack_topk):
        ratings = test_data[idx]
        items = test_negatives[idx]
        user = ratings[0]
        item = ratings[1]
        user_item_predict = {}
        items.append(item)
        if target_item is not None:
            items.append(target_item)
        users = np.full(len(items), user, dtype='int32')
        predictions = model.predict(
            [users, np.array(items)],
            batch_size = 512,
            verbose = 0
        )
        for j in range(len(items)):
            itm = items[j]
            user_item_predict[itm] = predictions[j]
        items.pop()
        ranklist = heapq.nlargest(attack_topk, user_item_predict, key=user_item_predict.get) # topk
        if target_item is not None:
            hr = self.get_hit_ratio(ranklist, target_item)
            ndcg = self.get_ndcg(ranklist, target_item)
            return hr, ndcg
        hr = self.get_hit_ratio(ranklist, item)
        ndcg = self.get_ndcg(ranklist, item)
        return hr, ndcg

    def get_hit_ratio(self, ranklist, item):
        for itm in ranklist:
            if itm == item:
                return 1
        return 0
    
    def get_ndcg(self, ranklist, item):
        for j in range(len(ranklist)):
            itm = ranklist[j]
            if itm == item:
                return math.log(2) / math.log(j + 2)
        return 0

if __name__ == '__main__':
    print('loading data from Dataset...')
    dataset = Dataset()
    num_users = dataset.num_users   
    num_items = dataset.num_items   
    X_train = dataset.X_train
    Y_train = dataset.Y_train 
    X_test= dataset.X_test
    Y_test = dataset.Y_test
    X_val = dataset.X_val
    Y_val = dataset.Y_val
    # target_items = dataset.target_items
    target_items = [3618, 2428, 3548, 3645, 3013, 3242, 3551, 3431, 3409, 3050]
    test_data = dataset.test_positives
    test_negatives = dataset.test_negatives

    max_n_attacker = 500
    eval_n_attacker = 90
    episode_length = 33

    print(f'target_items{target_items}')

    print('build model...')
    if os.path.exists('my_model'):
        model_layers = [64, 32, 16, 8]
        reg_mf = 0
        reg_mlp = [0, 0, 0, 0]
        k = 8
        learning_rate = 0.0005
        ncf = NeuMF(num_users+max_n_attacker, num_items, model_layers, reg_mf, reg_mlp, k)
        model = tf.keras.models.load_model("my_model")
    else:
        model_layers = [64, 32, 16, 8]
        reg_mf = 0
        reg_mlp = [0, 0, 0, 0]
        k = 8
        learning_rate = 0.0005
        ncf = NeuMF(num_users+max_n_attacker, num_items, model_layers, reg_mf, reg_mlp, k)
        model = ncf.get_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])
        print(model.summary())
        
        '''
        hits, ndcgs = ncf.evalueate_model(model, test_data, test_negatives)
        hr = np.array(hits).mean()
        ndcg = np.array(ndcgs).mean()
        '''

        print('start training...')
        batch_size = 512
        best_hr = 0
        best_ndcg= 0
        #history = LossHistory()

        history = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=30,
            validation_data=(X_val, Y_val),
            shuffle=True
        )
        model.save('my_model')
    '''
    hits, ndcgs = ncf.evalueate_model(model, test_data, test_negatives)
    hr = np.array(hits).mean()
    ndcg = np.array(ndcgs).mean()
    print('%d-th epoch: hr = %.3f, ndcg = %.3f' % (i, hr, ndcg))
    '''

    print('constrcuct attack profiles')
    target_item = target_items[0]
    attack_data = []
    for i in range(num_users, num_users+eval_n_attacker):
        attack_profile = [i]
        item_set = list(range(num_items))
        # j1 = random.sample(tmp, 1)[0]
        attack_profile.append(target_item)
        while(len(attack_profile)<=episode_length):
            j2 = random.sample(item_set,1)[0]
            attack_profile.append(j2)
            item_set.remove(j2)
        attack_data.append(attack_profile)
    X_train_attack, Y_train_attack = dataset.load_attack_data_get_attack_instances(attack_data)
    
    print(f'attack the item{target_item}')
    history = model.fit(
        X_train_attack,
        Y_train_attack,
        batch_size=512,
        epochs=10,
        validation_data=(X_val, Y_val),
        shuffle=True
    )
    print('evaluate attack performance: HR')
    idx = random.sample(list(range(len(test_data))), 50)
    spy_test_data = []
    spy_test_negatives = []
    for i in idx:
        spy_test_data.append(test_data[i])
        spy_test_negatives.append(test_negatives[i])
    hits, ndcgs = ncf.evaluate_model(model, spy_test_data, spy_test_negatives, target_item)
    hr = np.array(hits).mean()
    ndcg = np.array(ndcgs).mean()
    print('attack performance: hr = %.3f, ndcg = %.3f' % (hr, ndcg))


    # iterations=range(len(history.history['loss']))
    # plt.figure()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.plot(iterations, history.history['loss'], label='Training loss')
    # plt.plot(iterations, history.history['val_loss'], label='Validation loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.xlabel('epoch')
    # plt.ylabel('acc')
    # plt.plot(iterations, history.history['acc'], label='Training acc')
    # plt.plot(iterations, history.history['val_acc'], label='Validation acc')
    # plt.title('Traing and Validation acc')
    # plt.legend()
    # plt.show()

    '''
    
    '''