'''
Created on Aug. 09th 2019
Loading the data:
    1. training data:
        user-to-basket data: train_u2b.txt
        basket-to-item data: train_b2i.txt
    2. Testing data:
        basket-to-item data: test_b2i.txt
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file_u2b = path + '/train_u2b.txt'
        train_file_b2i = path + '/train_b2i.txt'
        test_file_b2i = path + '/test_b2i.txt'

        #get number of users, basket, items
        self.n_users, self.n_items, self.n_baskets = 0, 0, 0
        self.n_train_u2b, self.n_train_b2i, self.n_test_b2i = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []
        self.exist_baskets = []
            # load the number of users and basket
        with open(train_file_u2b) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    bids = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.exist_baskets.extend(bids)
                    self.n_baskets = max(self.n_baskets, max(bids))
                    self.n_users = max(self.n_users, uid)
                    self.n_train_u2b += len(bids)

            # load the number of items from train and test
        with open(train_file_b2i) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    iids = [int(i) for i in l[1:]]
                    bid = int(l[0])
                    self.n_items = max(self.n_items, max(iids))
                    self.n_train_b2i += len(iids)

        with open(test_file_b2i) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test_b2i += len(items)
        self.n_items += 1
        self.n_users += 1
        self.n_baskets += 1

        self.print_statistics()

        self.R_u2b = sp.dok_matrix((self.n_users, self.n_baskets), dtype=np.float32)
        self.R_b2i = sp.dok_matrix((self.n_baskets, self.n_items), dtype=np.float32)
        self.R_u2i = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_u2b, self.train_b2i, self.test_set, self.train_u2i = {}, {}, {}, {}
        self.train_u2i_prob = {}
        with open(train_file_u2b) as f:
            for line in f:
                l = line.strip().split()
                uid = int(l[0])
                baskets = [int(bid) for bid in l[1:]]
                for bid in baskets:
                    self.R_u2b[uid, bid] = 1.
                self.train_u2b[uid] = baskets

        with open(train_file_b2i) as f_train:
            with open(test_file_b2i) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    bid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R_b2i[bid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_b2i[bid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    bid, test_items = items[0], items[1:]
                    self.test_set[bid] = test_items
        self.b2u_dict = self.get_b2u()
        for uid in self.train_u2b:
            temp_bids = self.train_u2b[uid]
            user_items = set()
            for bid in temp_bids:
                temp_items = set(self.train_b2i[bid])
                user_items.union(temp_items)
            self.train_u2i[uid] = list(user_items)
        for uid in self.train_u2b:
            temp_bids = self.train_u2b[uid]
            item_prob = {}
            temp_total = 0
            for bid in temp_bids:
                temp_items = self.train_b2i[bid]
                temp_total += len(temp_items)
                for item in temp_items:
                    self.R_u2i[uid, item] = 1
                    if item in item_prob:
                        item_prob[item] += 1
                    else:
                        item_prob[item] = 1
            item_prob = {item:item_prob[item]/temp_total for item in item_prob}
            self.train_u2i_prob[uid] = item_prob



    def get_b2u(self):
        ret_dict = {}
        for bid in range(self.n_baskets):
            keys = self.R_u2b[:,bid].keys()
            for key in keys:
                uid = key[0]
            ret_dict[bid] = uid
        return ret_dict

    def get_corres_user(self, bids):
        uids = [self.b2u_dict[bid] for bid in bids]
        return uids

    def get_adj_mat_group(self):
        try:
            t1 = time()
            adj_mat_b2i = sp.load_npz(self.path + '/s_adj_mat_b2i.npz')
            norm_adj_mat_b2i = sp.load_npz(self.path + '/s_norm_adj_mat_b2i.npz')
            mean_adj_mat_b2i = sp.load_npz(self.path + '/s_mean_adj_mat_b2i.npz')
            adj_mat_u2b = sp.load_npz(self.path + '/s_adj_mat_u2b.npz')
            norm_adj_mat_u2b = sp.load_npz(self.path + '/s_norm_adj_mat_u2b.npz')
            mean_adj_mat_u2b = sp.load_npz(self.path + '/s_mean_adj_mat_u2b.npz')

            print('already load adj matrix', adj_mat_u2b.shape, time() - t1)

        except Exception:
            adj_mat_u2b, norm_adj_mat_u2b, mean_adj_mat_u2b = self.create_adj_mat_u2b()
            adj_mat_b2i, norm_adj_mat_b2i, mean_adj_mat_b2i = self.create_adj_mat_b2i()
            sp.save_npz(self.path + '/s_adj_mat_u2b.npz', adj_mat_u2b)
            sp.save_npz(self.path + '/s_norm_adj_mat_u2b.npz', norm_adj_mat_u2b)
            sp.save_npz(self.path + '/s_mean_adj_mat_u2b.npz', mean_adj_mat_u2b)
            sp.save_npz(self.path + '/s_adj_mat_b2i.npz', adj_mat_b2i)
            sp.save_npz(self.path + '/s_norm_adj_mat_b2i.npz', norm_adj_mat_b2i)
            sp.save_npz(self.path + '/s_mean_adj_mat_b2i.npz', mean_adj_mat_b2i)
        return adj_mat_u2b, norm_adj_mat_u2b, mean_adj_mat_u2b, adj_mat_b2i, norm_adj_mat_b2i, mean_adj_mat_b2i

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat_u2i = sp.load_npz(self.path + '/s_adj_mat_ubi_ui.npz')
            norm_adj_mat_u2i = sp.load_npz(self.path + '/s_norm_adj_mat_ubi_ui.npz')
            mean_adj_mat_u2i = sp.load_npz(self.path + '/s_mean_adj_mat_ubi_ui.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        except Exception:
            adj_mat_u2i, norm_adj_mat_u2i, mean_adj_mat_u2i = self.create_adj_mat_ubi_ui()
            sp.save_npz(self.path + '/s_adj_mat_ubi_ui.npz', adj_mat_u2i)
            sp.save_npz(self.path + '/s_norm_adj_mat_ubi_ui.npz', norm_adj_mat_u2i)
            sp.save_npz(self.path + '/s_mean_adj_mat_ubi_ui.npz', mean_adj_mat_u2i)
        return adj_mat_u2i, norm_adj_mat_u2i, mean_adj_mat_u2i

    def create_inter_mat(self, adj_type='norm'):
        R_u2b = self.R_u2b.todok()
        R_u2i = self.R_u2i.todok()
        R_b2i = self.R_b2i.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_lap(adj):
            rowsum = np.array(adj.sum(1))
            colsum = np.array(adj.sum(0))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            
            d_inv = np.power(colsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate laplacian norm adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'norm':
            R_u2b_norm = normalized_adj_single(R_u2b)
            R_u2b_t_norm = normalized_adj_single(R_u2b.T)
            R_u2i_norm = normalized_adj_single(R_u2i)
            R_u2i_t_norm = normalized_adj_single(R_u2i.T)
            R_b2i_norm = normalized_adj_single(R_b2i)
            R_b2i_t_norm = normalized_adj_single(R_b2i.T)
            return [R_u2b_norm.tocsr(), R_u2b_t_norm.tocsr(), R_u2i_norm.tocsr(), R_u2i_t_norm.tocsr(), R_b2i_norm.tocsr(), R_b2i_t_norm.tocsr()]
        elif adj_type == 'lap':
            R_u2b_norm = normalized_adj_lap(R_u2b)
            R_u2b_t_norm = normalized_adj_lap(R_u2b.T)
            R_u2i_norm = normalized_adj_lap(R_u2i)
            R_u2i_t_norm = normalized_adj_lap(R_u2i.T)
            R_b2i_norm = normalized_adj_lap(R_b2i)
            R_b2i_t_norm = normalized_adj_lap(R_b2i.T)
            return [R_u2b_norm.tocsr(), R_u2b_t_norm.tocsr(), R_u2i_norm.tocsr(), R_u2i_t_norm.tocsr(), R_b2i_norm.tocsr(), R_b2i_t_norm.tocsr()]
        else:
            return [R_u2b.tocsr(), R_u2b.T.tocsr(), R_u2i.tocsr(), R_u2i.T.tocsr(), R_b2i.tocsr(), R_b2i.T.tocsr()]

    def create_adj_mat_u2b(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_baskets, self.n_users + self.n_baskets), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R_u2b = self.R_u2b.tolil()

        adj_mat[:self.n_users, self.n_users:] = R_u2b
        adj_mat[self.n_users:, :self.n_users] = R_u2b.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix for user-basket', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def create_adj_mat_u2i(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R_u2i = self.R_u2i.tolil()

        adj_mat[:self.n_users, self.n_users:] = R_u2i
        adj_mat[self.n_users:, :self.n_users] = R_u2i.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix for user-item', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def create_adj_mat_b2i(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_baskets + self.n_items, self.n_baskets + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R_b2i = self.R_b2i.tolil()

        adj_mat[:self.n_baskets, self.n_baskets:] = R_b2i
        adj_mat[self.n_baskets:, :self.n_baskets] = R_b2i.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix for user-basket', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def create_adj_mat_ubi_ui(self):
        t1 = time()
        baskets_num = self.n_users + self.n_baskets
        total_num = self.n_users+ self.n_baskets + self.n_items
        adj_mat = sp.dok_matrix((total_num, total_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R_u2b = self.R_u2b.tolil()
        R_b2i = self.R_b2i.tolil()
        R_u2i = self.R_u2i.tolil()
        adj_mat[:self.n_users, self.n_users:baskets_num] = R_u2b
        adj_mat[self.n_users:baskets_num, :self.n_users] = R_u2b.T
        adj_mat[self.n_users:baskets_num, baskets_num:] = R_b2i
        adj_mat[baskets_num:, self.n_users:baskets_num] = R_b2i.T
        adj_mat[:self.n_users, baskets_num:] = R_u2i
        adj_mat[baskets_num:, :self.n_users] = R_u2i.T
        adj_mat = adj_mat.todok()
        t2 = time()
        print('already create adjacency matrix for user-basket-item', adj_mat.shape, t2 - t1)
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def negative_pool(self):
        t1 = time()
        for u in self.train_b2i.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_b2i[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        # this is the sample for basket
        if self.batch_size <= self.n_baskets:
            baskets = rd.sample(self.exist_baskets, self.batch_size)
        else:
            baskets = [rd.choice(self.exist_baskets) for _ in range(self.batch_size)]


        def sample_pos_items_for_basket(bid, num):
            pos_items = self.train_b2i[bid]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pos_items_for_basket_from_u(bid, num):
            uid = self.b2u_dict[bid]
            pos_prob = self.train_u2i_prob[uid].items()
            pos_items = [int(item[0]) for item in pos_prob]
            prob = [float(item[1]) for item in pos_prob]
            pos_batch = np.random.choice(pos_items, num, replace=False, p=prob)
            return list(pos_batch)

        def sample_neg_items_for_basket(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_b2i[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_basket_from_u(bid,num):
            neg_items = []
            uid = self.b2u_dict[bid]
            while  True:
                if len(neg_items) == num:break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_u2i and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_b2i[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for bid in baskets:
            pos_items += sample_pos_items_for_basket(bid, 1)
            neg_items += sample_neg_items_for_basket(bid, 1)

        return baskets, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items
    def get_num_basket(self):
        return self.n_baskets
    def get_num_item(self):
        return self.n_items
    def get_num_user(self):
        return self.n_users

    def print_statistics(self):
        print('n_users=%d, n_items=%d, n_baskets=%d' % (self.n_users, self.n_items, self.n_baskets))
        print('n_b2i_interactions=%d' % (self.n_train_b2i + self.n_test_b2i))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train_b2i, self.n_test_b2i, (self.n_train_b2i + self.n_test_b2i)/(self.n_baskets * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
