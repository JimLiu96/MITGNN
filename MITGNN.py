'''
Created on Nov. 04th 2019
Tensorflow Implementation of intent graph convolutional neural network model for basket recommendation
'''
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from utility.helper import *
from utility.batch_test_uAtt import *
import pickle
import numpy as np

class MITGNN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.num_intent = args.num_intent

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_baskets = data_config['n_baskets']

        self.n_fold = 100

        # self.norm_adj_u2b = data_config['norm_adj_u2b']
        # self.norm_adj_b2i = data_config['norm_adj_b2i']
        # self.norm_adj_ubi = data_config['norm_adj_ubi']
        # self.n_nonzero_elems = self.norm_adj_u2b.count_nonzero()
        self.inter_mat = data_config['inter_mat']
        for key in self.inter_mat:
            print('shape of ' + key + ' :', self.inter_mat[key].shape)
        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        # self.users = tf.placeholder(tf.int32, shape=(None,)) # satr
        self.baskets = tf.placeholder(tf.int32, shape=(None, ), name='input_baskets')
        self.pos_items = tf.placeholder(tf.int32, shape=(None,), name= 'pos_items')
        self.users = tf.placeholder(tf.int32, shape=(None, ), name = 'users')
        self.c_users = tf.placeholder(tf.int32, shape=(None, ), name='basket_corresponding_users')
        self.neg_items = tf.placeholder(tf.int32, shape=(None,), name='neg_items')

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None], name='node_dropout')
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None], name='message_dropout')

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['intent_conv']:
            self.ua_embeddings, self.ba_embeddings, self.ia_embeddings = self._create_intent_conv()
        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()
        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()
        elif self.alg_type in ['rgcn']:
            self.ua_embeddings, self.ba_embeddings, self.ia_embeddings = self._create_rgcn_embed()
        elif self.alg_type in ['intent_conv_plus']:
            self.ua_embeddings, self.ba_embeddings, self.ia_embeddings = self._create_intent_conv_plus()
        elif self.alg_type in ['intent_conv_att']:
            self.ua_embeddings, self.ba_embeddings, self.ia_embeddings = self._create_intent_conv_att()
        elif self.alg_type in ['intent_conv_att_no_inter']:
            self.ua_embeddings, self.ba_embeddings, self.ia_embeddings = self._create_intent_conv_att_no_inter()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        # self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        # self.b_g_embeddings = tf.nn.embedding_lookup(self.ba_embeddings_b2i, self.baskets)
        # self.b_g_embeddings = [0]*self.num_intent
        self.b_over_embeddings = tf.reduce_sum(self.ba_embeddings, 0)
        self.b_g_embeddings = tf.nn.embedding_lookup(self.b_over_embeddings, self.baskets)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.u_c_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.c_users)
        # basket user attention
        self.b_at_embeddings = self.u_c_embeddings

        """
        *********************************************************
        Inference for the testing phase.
        """
        # self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        self.batch_ratings = tf.matmul(self.b_at_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.b_at_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            all_weights['basket_embedding'] = [tf.Variable(tf.zeros([self.n_baskets, self.emb_dim], tf.float32), name='basket_embedding_'+str(k), trainable=True) for k in range(self.num_intent)]
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user'][:,0:self.emb_dim], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            # all_weights['basket_embedding'] = [tf.Variable(tf.zeros([self.n_baskets, self.emb_dim], tf.float32), name='basket_embedding_'+str(k), trainable=False) for k in range(self.num_intent)]
            all_weights['basket_embedding'] = [tf.Variable(initializer([self.n_baskets, self.emb_dim], tf.float32), name='basket_embedding_'+str(k), trainable=True) for k in range(self.num_intent)]
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item'][:,0:self.emb_dim], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained embedding initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size
        # user basket item convolutinal layer weights
        # for k in range(self.n_layers):
        #     all_weights['W_gc_%d' %k] = tf.Variable(
        #         initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
        #     all_weights['b_gc_%d' %k] = tf.Variable(
        #         initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

        #     all_weights['W_bi_%d' % k] = tf.Variable(
        #         initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
        #     all_weights['b_bi_%d' % k] = tf.Variable(
        #         initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)
        all_weights['att_left'] = tf.Variable(initializer([self.emb_dim,1]), name='att_left')
        all_weights['att_right'] = tf.Variable(initializer([self.emb_dim,1]), name='att_right')
        for k in range(self.n_layers):
            all_weights['W_self_user_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_self_user_%d' % k)
            all_weights['W_self_bas_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_self_bas_%d' % k)
            all_weights['W_self_item_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_self_item_%d' % k)
            all_weights['b_self_user_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_self_user_%d' % k)
            all_weights['b_self_bas_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_self_bas_%d' % k)
            all_weights['b_self_item_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_self_item_%d' % k)
            all_weights['W_ub_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_ub_%d' % k)
            all_weights['b_ub_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_ub_%d' % k)
            all_weights['W_ui_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_ui_%d' % k)
            all_weights['b_ui_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_ui_%d' % k)
            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)
            # the weights for intent
            # all_weights['W_ub_t_%d' %k] = tf.Variable(
            #     initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_ub_t_%d' % k)
            # all_weights['b_ub_t_%d' %k] = tf.Variable(
            #     initializer([1, self.weight_size_list[k+1]]), name='b_ub_t_%d' % k)
            # all_weights['W_bi_t_%d' % k] = tf.Variable(
            #     initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_t_%d' % k)
            # all_weights['b_bi_t_%d' % k] = tf.Variable(
            #     initializer([1, self.weight_size_list[k + 1]]), name='b_bi_t_%d' % k)
        # basket to item convolutonal layer weights
        # for k in range(self.n_layers):

        #     all_weights['W_bi_%d_b2i' % k] = tf.Variable(
        #         initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d_b2i' % k)
        #     all_weights['b_bi_%d_b2i' % k] = tf.Variable(
        #         initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d_b2i' % k)

        return all_weights

    def _split_A_hat_u2b(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_baskets) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_baskets
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_b2i(self, X):
        A_fold_hat = []

        fold_len = (self.n_baskets + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_baskets + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat(self, X):
        A_fold_hat = []
        total = X.shape[0]
        fold_len = total // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = total
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_intent_conv_att_no_inter(self):
        # intent conv with bias
        basket_embedding = [self.weights['basket_embedding']]
        item_embedding = [self.weights['item_embedding']]
        user_embedding = [self.weights['user_embedding']]
        # all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            u2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b_t']), user_embedding[k])
        # print('u2b_embedding shape:',u2b_embedding.shape)
            i2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i']), item_embedding[k])
            temp_basket_embedding = [0] * self.num_intent
            for t in range(self.num_intent):
                temp_basket_embedding[t] = tf.nn.leaky_relu(tf.matmul(u2b_embedding, self.weights['W_ub_%d' %k])+self.weights['b_ub_%d' %k]) 
                temp_basket_embedding[t] += tf.nn.leaky_relu(tf.matmul(i2b_embedding, self.weights['W_bi_%d' %k])+self.weights['b_bi_%d' %k])
            temp_embedding_att_u2b =  [tf.nn.leaky_relu(tf.matmul(temp_embed_, self.weights['att_left']) 
                                            + tf.matmul(u2b_embedding, self.weights['att_right'])) for temp_embed_ in temp_basket_embedding]   
            basket_user_att_embedding = tf.zeros(tf.shape(basket_embedding[k][0]))
            for t in range(self.num_intent):
                basket_user_att_embedding = tf.multiply(tf.tile(temp_embedding_att_u2b[t], [1,self.emb_dim]), basket_embedding[k][t])
            b2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b']), basket_user_att_embedding)
            i2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i']), item_embedding[k])
            temp_user_embedding = tf.nn.leaky_relu(tf.matmul(b2u_embedding, self.weights['W_ub_%d' %k]) + self.weights['b_ub_%d' %k])
            temp_user_embedding += tf.nn.leaky_relu(tf.matmul(i2u_embedding, self.weights['W_ui_%d' %k])+ self.weights['b_ui_%d' %k])
            basket_item_att_embedding = tf.zeros(tf.shape(basket_embedding[k][0]))
            temp_embedding_att_i2b =  [tf.nn.leaky_relu(tf.matmul(temp_embed_, self.weights['att_left']) 
                                            + tf.matmul(i2b_embedding, self.weights['att_right'])) for temp_embed_ in temp_basket_embedding]
            for t in range(self.num_intent):
                basket_item_att_embedding = tf.multiply(tf.tile(temp_embedding_att_i2b[t], [1,self.emb_dim]), basket_embedding[k][t])
            b2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i_t']), basket_item_att_embedding)
            u2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i_t']), user_embedding[k])
            temp_item_embedding = tf.nn.leaky_relu(tf.matmul(b2i_embedding, self.weights['W_bi_%d' %k]) + self.weights['b_bi_%d' %k]) 
            temp_item_embedding += tf.nn.leaky_relu(tf.matmul(item_embedding[k], self.weights['W_ui_%d' %k]) + self.weights['b_ui_%d' %k])
            temp_basket_e_norm = [0] * self.num_intent
            for t in range(self.num_intent):
                temp_basket_e_norm[t] = tf.math.l2_normalize(temp_basket_embedding[t])
                temp_basket_e_norm[t] = tf.nn.dropout(temp_basket_e_norm[t], 1 - self.mess_dropout[k])
            temp_user_e_norm = tf.math.l2_normalize(temp_user_embedding)
            temp_user_e_norm = tf.nn.dropout(temp_user_e_norm, 1 - self.mess_dropout[k])
            temp_item_e_norm = tf.math.l2_normalize(temp_item_embedding)
            temp_item_e_norm = tf.nn.dropout(temp_item_e_norm, 1 - self.mess_dropout[k])

            basket_embedding += [temp_basket_e_norm]
            item_embedding += [temp_item_e_norm]
            user_embedding += [temp_user_e_norm]
        # all_embeddings = tf.concat(all_embeddings, 1)
        # u_g_embeddings = tf.concat(user_embedding, 1)
        u_g_embeddings = tf.reduce_mean(basket_embedding, 1)
        # basket_layer = []
        b_g_embeddings = [0] * self.num_intent
        for t in range(self.num_intent):
            basket_layer = [basket_embedding[k][t] for k in range(0,self.n_layers+1)]
            # print(len(basket_layer))
            b_g_embeddings[t] = tf.reduce_mean(basket_layer, 1)
            basket_layer = []
            # tf.print(tf.shape(b_g_embeddings[t]))
        # b_g_embeddings = tf.concat(basket_embedding, 1)
        # i_g_embeddings = tf.concat(item_embedding, 1)
        i_g_embeddings = tf.reduce_mean(item_embedding, 1)
        # u_g_embeddings, b_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_baskets, self.n_items], 0)
        return u_g_embeddings, b_g_embeddings, i_g_embeddings

    def _create_intent_conv_att(self):
        # intent conv with bias
        basket_embedding = [self.weights['basket_embedding']]
        item_embedding = [self.weights['item_embedding']]
        user_embedding = [self.weights['user_embedding']]
        # all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            u2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b_t']), user_embedding[k])
        # print('u2b_embedding shape:',u2b_embedding.shape)
            i2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i']), item_embedding[k])
            temp_basket_embedding = [0] * self.num_intent
            for t in range(self.num_intent):
                temp_basket_embedding[t] = tf.nn.leaky_relu(tf.matmul(basket_embedding[k][t], self.weights['W_self_bas_%d' %k])+self.weights['b_self_bas_%d' %k]) 
                temp_basket_embedding[t] += tf.nn.leaky_relu(tf.matmul(tf.add(u2b_embedding, basket_embedding[k][t]), self.weights['W_ub_%d' %k])+self.weights['b_ub_%d' %k]) 
                temp_basket_embedding[t] += tf.nn.leaky_relu(tf.matmul(tf.add(i2b_embedding, basket_embedding[k][t]), self.weights['W_bi_%d' %k])+self.weights['b_bi_%d' %k])
            temp_embedding_att_u2b =  [tf.nn.leaky_relu(tf.matmul(temp_embed_, self.weights['att_left']) 
                                            + tf.matmul(u2b_embedding, self.weights['att_right'])) for temp_embed_ in temp_basket_embedding]   
            basket_user_att_embedding = tf.zeros(tf.shape(basket_embedding[k][0]))
            for t in range(self.num_intent):
                basket_user_att_embedding = tf.math.add(basket_user_att_embedding, tf.multiply(tf.tile(temp_embedding_att_u2b[t], [1,self.emb_dim]), basket_embedding[k][t]))
            b2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b']), basket_user_att_embedding)
            i2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i']), item_embedding[k])
            temp_user_embedding = tf.nn.leaky_relu(tf.matmul(user_embedding[k], self.weights['W_self_user_%d' %k])+self.weights['b_self_user_%d' %k]) 
            temp_user_embedding += tf.nn.leaky_relu(tf.matmul(tf.add(b2u_embedding, user_embedding[k]), self.weights['W_ub_%d' %k]) + self.weights['b_ub_%d' %k])
            temp_user_embedding += tf.nn.leaky_relu(tf.matmul(tf.add(i2u_embedding, user_embedding[k]), self.weights['W_ui_%d' %k])+ self.weights['b_ui_%d' %k])
            basket_item_att_embedding = tf.zeros(tf.shape(basket_embedding[k][0]))
            temp_embedding_att_i2b =  [tf.nn.leaky_relu(tf.matmul(temp_embed_, self.weights['att_left']) 
                                            + tf.matmul(i2b_embedding, self.weights['att_right'])) for temp_embed_ in temp_basket_embedding]
            for t in range(self.num_intent):
                basket_item_att_embedding = tf.math.add(basket_item_att_embedding, tf.multiply(tf.tile(temp_embedding_att_i2b[t], [1,self.emb_dim]), basket_embedding[k][t]))
            b2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i_t']), basket_item_att_embedding)
            u2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i_t']), user_embedding[k])
            temp_item_embedding = tf.nn.leaky_relu(tf.matmul(item_embedding[k], self.weights['W_self_item_%d' %k]) + self.weights['b_self_item_%d' %k])
            temp_item_embedding += tf.nn.leaky_relu(tf.matmul(tf.add(b2i_embedding, item_embedding[k]), self.weights['W_bi_%d' %k]) + self.weights['b_bi_%d' %k]) 
            temp_item_embedding += tf.nn.leaky_relu(tf.matmul(tf.add(u2i_embedding, item_embedding[k]), self.weights['W_ui_%d' %k]) + self.weights['b_ui_%d' %k])
            temp_basket_e_norm = [0] * self.num_intent
            for t in range(self.num_intent):
                temp_basket_e_norm[t] = tf.math.l2_normalize(temp_basket_embedding[t])
                temp_basket_e_norm[t] = tf.nn.dropout(temp_basket_e_norm[t], 1 - self.mess_dropout[k])
            temp_user_e_norm = tf.math.l2_normalize(temp_user_embedding)
            temp_user_e_norm = tf.nn.dropout(temp_user_e_norm, 1 - self.mess_dropout[k])
            temp_item_e_norm = tf.math.l2_normalize(temp_item_embedding)
            temp_item_e_norm = tf.nn.dropout(temp_item_e_norm, 1 - self.mess_dropout[k])

            basket_embedding += [temp_basket_e_norm]
            item_embedding += [temp_item_e_norm]
            user_embedding += [temp_user_e_norm]
        # all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings = tf.concat(user_embedding, 1)
        # basket_layer = []
        b_g_embeddings = [0] * self.num_intent
        for t in range(self.num_intent):
            basket_layer = [basket_embedding[k][t] for k in range(0,self.n_layers+1)]
            print(len(basket_layer))
            b_g_embeddings[t] = tf.concat(basket_layer, 1)
            basket_layer = []
            # tf.print(tf.shape(b_g_embeddings[t]))
        # b_g_embeddings = tf.concat(basket_embedding, 1)
        i_g_embeddings = tf.concat(item_embedding, 1)
        # u_g_embeddings, b_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_baskets, self.n_items], 0)
        return u_g_embeddings, b_g_embeddings, i_g_embeddings


    def _create_intent_conv(self):
        # intent conv with bias
        basket_embedding = [self.weights['basket_embedding']]
        item_embedding = [self.weights['item_embedding']]
        user_embedding = [self.weights['user_embedding']]
        # all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            u2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b_t']), user_embedding[k])
        # print('u2b_embedding shape:',u2b_embedding.shape)
            i2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i']), item_embedding[k])
            temp_basket_embedding = [0] * self.num_intent
            for t in range(self.num_intent):
                temp_basket_embedding[t] = tf.nn.leaky_relu(tf.matmul(basket_embedding[k][t], self.weights['W_self_bas_%d' %k])+self.weights['b_self_bas_%d' %k]) 
                temp_basket_embedding[t] += tf.nn.leaky_relu(tf.matmul(tf.multiply(u2b_embedding, basket_embedding[k][t]), self.weights['W_ub_%d' %k])+self.weights['b_ub_%d' %k]) 
                temp_basket_embedding[t] += tf.nn.leaky_relu(tf.matmul(tf.multiply(i2b_embedding, basket_embedding[k][t]), self.weights['W_bi_%d' %k])+self.weights['b_bi_%d' %k])
            basket_user_att_embedding = tf.zeros(tf.shape(basket_embedding[k][0]))
            for t in range(self.num_intent):
                basket_user_att_embedding = tf.math.add(basket_user_att_embedding, tf.multiply(u2b_embedding, basket_embedding[k][t]))
            b2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b']), basket_user_att_embedding)
            i2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i']), item_embedding[k])
            temp_user_embedding = tf.nn.leaky_relu(tf.matmul(user_embedding[k], self.weights['W_self_user_%d' %k])+self.weights['b_self_user_%d' %k]) 
            temp_user_embedding += tf.nn.leaky_relu(tf.matmul(tf.multiply(b2u_embedding, user_embedding[k]), self.weights['W_ub_%d' %k]) + self.weights['b_ub_%d' %k])
            temp_user_embedding += tf.nn.leaky_relu(tf.matmul(tf.multiply(i2u_embedding, user_embedding[k]), self.weights['W_ui_%d' %k])+ self.weights['b_ui_%d' %k])
            basket_item_att_embedding = tf.zeros(tf.shape(basket_embedding[k][0]))
            for t in range(self.num_intent):
                basket_item_att_embedding = tf.math.add(basket_item_att_embedding, tf.multiply(i2b_embedding, basket_embedding[k][t]))
            b2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i_t']), basket_item_att_embedding)
            u2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i_t']), user_embedding[k])
            temp_item_embedding = tf.nn.leaky_relu(tf.matmul(item_embedding[k], self.weights['W_self_item_%d' %k]) + self.weights['b_self_item_%d' %k])
            temp_item_embedding += tf.nn.leaky_relu(tf.matmul(tf.multiply(b2i_embedding, item_embedding[k]), self.weights['W_bi_%d' %k]) + self.weights['b_bi_%d' %k]) 
            temp_item_embedding += tf.nn.leaky_relu(tf.matmul(tf.multiply(u2i_embedding, item_embedding[k]), self.weights['W_ui_%d' %k]) + self.weights['b_ui_%d' %k])
            temp_basket_e_norm = [0] * self.num_intent
            for t in range(self.num_intent):
                temp_basket_e_norm[t] = tf.math.l2_normalize(temp_basket_embedding[t])
                temp_basket_e_norm[t] = tf.nn.dropout(temp_basket_e_norm[t], 1 - self.mess_dropout[k])
            temp_user_e_norm = tf.math.l2_normalize(temp_user_embedding)
            temp_user_e_norm = tf.nn.dropout(temp_user_e_norm, 1 - self.mess_dropout[k])
            temp_item_e_norm = tf.math.l2_normalize(temp_item_embedding)
            temp_item_e_norm = tf.nn.dropout(temp_item_e_norm, 1 - self.mess_dropout[k])

            basket_embedding += [temp_basket_e_norm]
            item_embedding += [temp_item_e_norm]
            user_embedding += [temp_user_e_norm]
        # all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings = tf.concat(user_embedding, 1)
        # basket_layer = []
        b_g_embeddings = [0] * self.num_intent
        for t in range(self.num_intent):
            basket_layer = [basket_embedding[k][t] for k in range(0,self.n_layers+1)]
            print(len(basket_layer))
            b_g_embeddings[t] = tf.concat(basket_layer, 1)
            basket_layer = []
            # tf.print(tf.shape(b_g_embeddings[t]))
        # b_g_embeddings = tf.concat(basket_embedding, 1)
        i_g_embeddings = tf.concat(item_embedding, 1)
        # u_g_embeddings, b_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_baskets, self.n_items], 0)
        return u_g_embeddings, b_g_embeddings, i_g_embeddings

    def _create_ngcf_embed_ubi(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_ubi)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['basket_embedding'],
                                     self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum 
            side_embeddings = tf.concat(temp_embed, 0)
            print('side_embeddings:', side_embeddings.shape)
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, b_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_baskets, self.n_items], 0)
        return u_g_embeddings, b_g_embeddings, i_g_embeddings

    def _create_rgcn_embed(self):
        basket_embedding = [self.weights['basket_embedding']]
        item_embedding = [self.weights['item_embedding']]
        user_embedding = [self.weights['user_embedding']]
        for k in range(0, self.n_layers):
            u2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b_t']), user_embedding[k])
            i2b_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i']), item_embedding[k])
            temp_basket_embedding = tf.nn.leaky_relu(tf.matmul(basket_embedding[k], self.weights['W_self_bas_%d' %k])+self.weights['b_self_bas_%d' %k])
            temp_basket_embedding += tf.nn.leaky_relu(tf.matmul(u2b_embedding, self.weights['W_ub_%d' %k])+self.weights['b_ub_%d' %k])
            temp_basket_embedding += tf.nn.leaky_relu(tf.matmul(i2b_embedding, self.weights['W_bi_%d' %k]) + self.weights['b_bi_%d' %k])
            b2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2b']), basket_embedding[k])
            i2u_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i']), item_embedding[k])
            temp_user_embedding = tf.nn.leaky_relu(tf.matmul(user_embedding[k], self.weights['W_self_user_%d' %k])+self.weights['b_self_user_%d' %k]) 
            temp_user_embedding += tf.nn.leaky_relu(tf.matmul(b2u_embedding, self.weights['W_ub_%d' %k]) + self.weights['b_ub_%d' %k])
            temp_user_embedding += tf.nn.leaky_relu(tf.matmul(i2u_embedding, self.weights['W_ui_%d' %k]) + self.weights['b_ui_%d' %k])
            b2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['b2i_t']), basket_embedding[k])
            u2i_embedding = tf.sparse_tensor_dense_matmul(self._convert_sp_mat_to_sp_tensor(self.inter_mat['u2i_t']), user_embedding[k])
            temp_item_embedding = tf.nn.leaky_relu(tf.matmul(item_embedding[k], self.weights['W_self_item_%d' %k]) + self.weights['b_self_item_%d' %k])
            temp_item_embedding += tf.nn.leaky_relu(tf.matmul(tf.multiply(b2i_embedding, item_embedding[k]), self.weights['W_bi_%d' %k]) + self.weights['b_bi_%d' %k])
            temp_item_embedding += tf.nn.leaky_relu(tf.matmul(tf.multiply(u2i_embedding, item_embedding[k]), self.weights['W_ui_%d' %k]) + self.weights['b_ui_%d' %k])
            basket_embedding += [temp_basket_embedding]
            item_embedding += [temp_item_embedding]
            user_embedding += [temp_user_embedding]
        u_g_embeddings = tf.concat(user_embedding, 1)
        b_g_embeddings = tf.concat(basket_embedding, 1)
        i_g_embeddings = tf.concat(item_embedding, 1)
        return u_g_embeddings, b_g_embeddings, i_g_embeddings


    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores) + 0.0001)
        mf_loss = tf.negative(tf.reduce_mean(maxi))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embeddings')
    print(pretrain_path)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
        for key in pretrain_data:
            print(pretrain_data[key].shape)
        # print(pretrain_data['user'][0:10,:64])
    except Exception:
        print('cannot load pretrained embeddings !!!')
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_baskets'] = data_generator.n_baskets

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    inter_mat = dict()
    # plain_adj_u2b, norm_adj_u2b, mean_adj_u2b, plain_adj_b2i, norm_adj_b2i, mean_adj_b2i = data_generator.get_adj_mat()
    # plain_adj_ubi, norm_adj_ubi, mean_adj_ubi = data_generator.get_adj_mat()
    adj_mat = data_generator.create_inter_mat(adj_type=args.adj_type)
    inter_mat['u2b'] = adj_mat[0]
    inter_mat['u2b_t'] = adj_mat[1]
    inter_mat['u2i'] = adj_mat[2]
    inter_mat['u2i_t'] = adj_mat[3]
    inter_mat['b2i'] = adj_mat[4]
    inter_mat['b2i_t'] = adj_mat[5]
    # plain_adj_b2i, norm_adj_b2i, mean_adj_b2i = data_generator.get_adj_mat()
    # if args.adj_type == 'plain':
    #     config['norm_adj_ubi'] = plain_adj_ubi
    #     print('use the plain adjacency matrix')

    # elif args.adj_type == 'norm':
    #     config['norm_adj_ubi'] = norm_adj_ubi
    #     print('use the normalized adjacency matrix')

    # elif args.adj_type == 'gcmc':
    #     config['norm_adj_ubi'] = mean_adj_ubi
    #     print('use the gcmc adjacency matrix')

    # else:
    #     config['norm_adj_ubi'] = mean_adj_ubi + sp.eye(mean_adj_ubi.shape[0])
    #     print('use the mean adjacency matrix')
    config['inter_mat'] = inter_mat
    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = MITGNN(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    # """
    # if args.report == 1:
    #     assert args.test_flag == 'full'
    #     users_to_test_list, split_state = data_generator.get_sparsity_split()
    #     users_to_test_list.append(list(data_generator.test_set.keys()))
    #     split_state.append('all')

    #     report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    #     ensureDir(report_path)
    #     f = open(report_path, 'w')
    #     f.write(
    #         'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
    #         % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

    #     for i, users_to_test in enumerate(users_to_test_list):
    #         ret = test(sess, model, users_to_test, drop_flag=True)

    #         final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
    #                      ('\t'.join(['%.5f' % r for r in ret['recall']]),
    #                       '\t'.join(['%.5f' % r for r in ret['precision']]),
    #                       '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
    #                       '\t'.join(['%.5f' % r for r in ret['ndcg']]))
    #         print(final_perf)

    #         f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
    #     f.close()
    #     exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train_b2i // args.batch_size + 1

        for idx in range(n_batch):
            baskets, pos_items, neg_items = data_generator.sample()
            c_users = data_generator.get_corres_user(baskets)
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.baskets: baskets, model.c_users:c_users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})
            # print(batch_loss)
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        # if (epoch + 1) % 10 != 0:
        #     if args.verbose > 0 and epoch % args.verbose == 0:
        #         perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
        #             epoch, time() - t1, loss, mf_loss, reg_loss)
        #         print(perf_str)
        #     continue

        t2 = time()
        # users_to_test = list(data_generator.test_set.keys())
        print('Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, t2 - t1, loss, mf_loss, emb_loss, reg_loss))
        baskets_to_test = list(data_generator.test_set.keys())
        if (epoch==0 or (epoch > 50 and epoch%10==0) or (epoch > args.test_epoch and epoch%5==0)):
            ret = test(sess, model, baskets_to_test, drop_flag=True)

            t3 = time()
            print('Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, t2 - t1, t3-t2, loss, mf_loss, emb_loss, reg_loss))
            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            if args.verbose > 0:
                perf_str = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                print(perf_str)

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][-1], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step=200)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            # if should_stop == True:
            #     break

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['recall'][-1] == cur_best_pre_0:
                print('--------------best------------------')  
            if args.save_flag == 1:
                save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                print('save the weights in path: ', weights_save_path)
                u_embed, b_embed, i_embed = sess.run([model.u_g_embeddings, model.b_g_embeddings, model.pos_i_g_embeddings], 
                    feed_dict={model.baskets:list(range(model.n_baskets)), model.users:list(range(model.n_users)), 
                    model.pos_items:list(range(model.n_items)), model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.]*len(eval(args.layer_size))})
                embed_file_name = weights_save_path + '/embeddings.npz'
                np.savez(embed_file_name, user=u_embed, basket=b_embed, item=i_embed)
                print('save the embeddings in path:',embed_file_name)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
    sess.close()
