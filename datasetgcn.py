from random import sample 
import time
import numpy as np
from global_parameters import *
import networkx as nx
from scipy.sparse import csr_matrix
uu_reverse = True
ratio = 1#0.5
np.random.seed(1234)

class Data_loader():
    def __init__(self, train_u,train_v,train_r,\
        test_u, test_v,test_r,
        history_u_lists, history_ur_lists,history_v_lists, history_vr_lists, social_adj_lists,rating_list,gcn=False):

        self.train_u = train_u
        self.train_v = train_v
        self.train_label = train_r
        self.rating_list = []
        for rating in rating_list:
            self.rating_list.append(rating)
        
        self.test_u,self.test_v,self.test_label = test_u,test_v,test_r

        self.num_user = len(history_u_lists)
        self.num_item = len(history_v_lists)
        
        self.num_train_pair = len(self.train_u)
        self.num_test_pair = len(self.test_u)

        uv_rows = {}
        uv_cols = {}
        uv_datas= {}

        vu_rows = {}
        vu_cols = {}
        vu_datas= {}

        uu_row =  []
        uu_col =  []
        uu_data=  []

        # 
        uv_row = []
        uv_col = []
        uv_data= []

        vu_row = []
        vu_col = []
        vu_data= []

        uvu_row =  []
        uvu_col =  []
        uvu_data=  []

        vuv_row =  []
        vuv_col =  []
        vuv_data=  []




        for rating in rating_list:
            uv_rows[rating] = []
            uv_cols[rating] = []
            uv_datas[rating] = []

            vu_rows[rating] = []
            vu_cols[rating] = []
            vu_datas[rating] = []


        for uid in range(self.num_user):
            if uid in social_adj_lists:
                for uid2 in (social_adj_lists[uid]):
                    if uu_reverse:    
                        uu_row.append(uid2)
                        uu_col.append(uid)
                        uu_data.append(1.0/len(social_adj_lists[uid2]))
                    else:
                        uu_row.append(uid)
                        uu_col.append(uid2)
                        uu_data.append(1.0/len(social_adj_lists[uid]))
        
        self.uu_adj = csr_matrix((uu_data, (uu_row, uu_col)), shape=(self.num_user,self.num_user))

        if gcn:
            uu_row =  []
            uu_col =  []
            uu_data=  []

            undirected_social_adj_lists = {}
            for uid in range(self.num_user):
                undirected_social_adj_lists[uid]=set()
                undirected_social_adj_lists[uid].add(uid)

            for uid in range(self.num_user):
                for uid2 in social_adj_lists[uid]:
                    undirected_social_adj_lists[uid].add(uid2)
                    undirected_social_adj_lists[uid2].add(uid)
            self.uu_adj_gcn = csr_matrix((uu_data, (uu_row, uu_col)), shape=(self.num_user,self.num_user))

            undirected_uvu = {}
            for uid in range(self.num_user):
                undirected_uvu[uid]=set()
                undirected_uvu[uid].add(uid)

            for uid in range(self.num_user):
                for vid in history_u_lists[uid]:
                    for uid2 in history_v_lists[vid]:
                        undirected_uvu[uid].add(uid2)
                        undirected_uvu[uid2].add(uid)
                   #      if len(undirected_uvu[uid])>30:
                   #          break
                   # if len(undirected_uvu[uid])>30:
                   #          break

            for uid in range(self.num_user):
                for uid2 in (undirected_uvu[uid]):
                    uvu_row.append(uid2)
                    uvu_col.append(uid)
                    uvu_data.append(1.0/len(undirected_uvu[uid2]))
            self.uvu_adj = csr_matrix((uvu_data, (uvu_row, uvu_col)), shape=(self.num_user,self.num_user))

            print('create uvu done')

            undirected_vuv = {}
            for vid in range(self.num_item):
                undirected_vuv[vid]=set()
                undirected_vuv[vid].add(vid)

            for vid in range(self.num_item):
                for uid in history_v_lists[vid]:
                    for vid2 in history_u_lists[uid]:
                        undirected_vuv[vid].add(vid2)
                        undirected_vuv[vid2].add(vid)
                   #      if len(undirected_vuv[vid])>20:
                   #          break
                   # if len(undirected_vuv[vid])>20:
                   #          break


            for vid in range(self.num_item):
                for vid2 in (undirected_vuv[vid]):
                    vuv_row.append(vid2)
                    vuv_col.append(vid)
                    vuv_data.append(1.0/len(undirected_vuv[vid2]))
            self.vuv_adj = csr_matrix((vuv_data, (vuv_row, vuv_col)), shape=(self.num_item,self.num_item))
            print('create vuv done')


        for uid in range(self.num_user):
            for vid,rating in list(zip(history_u_lists[uid],history_ur_lists[uid])):
                if rating in rating_list:
                    uv_rows[rating].append(uid)
                    uv_cols[rating].append(vid)
                    uv_datas[rating].append(1.0/len(history_v_lists[vid]))

                    uv_row.append(uid)
                    uv_col.append(vid)
                    # uv_data.append(1.0/len(history_v_lists[vid]))
                    uv_data.append(1.0)


        for vid in range(self.num_item):
            for uid,rating in list(zip(history_v_lists[vid],history_vr_lists[vid])):
                if rating in rating_list:
                    vu_rows[rating].append(vid)
                    vu_cols[rating].append(uid)
                    vu_datas[rating].append(1.0/len(history_u_lists[uid]))
                    # vu_datas[rating].append(1.0/len(history_u_lists[uid]))

                    vu_row.append(vid)
                    vu_col.append(uid)
                    # vu_data.append(1.0/len(history_u_lists[uid]))
                    vu_data.append(1.0)


        self.uv_adj = csr_matrix((uv_data, (uv_row, uv_col)), shape=(self.num_user,self.num_item))
        self.vu_adj = csr_matrix((vu_data, (vu_row, vu_col)), shape=(self.num_item,self.num_user))



        self.uv_adjs = {}
        self.vu_adjs = {}
        for rating in rating_list:

            # print(rating,len(uv_datas[rating]))
            print(rating,len(vu_datas[rating]))

            self.uv_adjs[rating] = csr_matrix((uv_datas[rating], (uv_rows[rating], uv_cols[rating])), shape=(self.num_user,self.num_item))
            self.vu_adjs[rating] = csr_matrix((vu_datas[rating], (vu_rows[rating], vu_cols[rating])), shape=(self.num_item,self.num_user))
        

        if ratio<1:
            print('shrink train')
            num_pair = int(ratio*self.num_train_pair)
            self.num_train_pair = num_pair
            u,v,r = self.train_u,self.train_v,self.train_label
            uvr = np.stack([u,v,r],axis=-1)
            np.random.shuffle(uvr)
            self.train_u = uvr[:num_pair,0]
            self.train_v = uvr[:num_pair,1]
            self.train_label = uvr[:num_pair,2]


    def  get_test(self):
        return self.num_test_pair,self.test_u,self.test_v,self.test_label

    def  get_valid(self):
        num_pair = self.num_test_pair//2
        return num_pair,self.test_u[:num_pair],self.test_v[:num_pair],self.test_label[:num_pair]

    def  get_valid_test(self):
        num_pair = self.num_test_pair//2
        return self.num_test_pair-num_pair,self.test_u[num_pair:],self.test_v[num_pair:],self.test_label[num_pair:]


    def construct_adj(self, adj_list, max_degree):

        adj = len(adj_list)*np.ones((len(adj_list)+1, max_degree))
        deg = np.zeros((len(adj_list),))

        for index in range(len(adj_list)):
            neighbors = np.array(adj_list[index])
            deg[index] = len(neighbors)

            if len(neighbors) == 0:
                print('maybe a bug here!!!!!!!!!',index)
                continue

            if len(neighbors) > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif len(neighbors) < max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=True)
            adj[index, :] = neighbors
        return adj, deg



    def get_batch(self,train= True, batch_size = 16384):

        # sample

        if train:
            num_pair = self.num_train_pair
            u,v,r = self.train_u,self.train_v,self.train_label
            uvr = np.stack([u,v,r],axis=-1)
            np.random.shuffle(uvr)
            u = uvr[:,0]
            v = uvr[:,1]
            r = uvr[:,2]
        else:
            num_pair = self.num_test_pair
            u,v,r = self.test_u,self.test_v,self.test_label

        batch_begin = 0
        batch_end = batch_size+batch_begin
        if batch_end> num_pair:
            batch_end= num_pair    
            
        while batch_begin<num_pair:
            batch_u = u[batch_begin:batch_end]
            batch_v = v[batch_begin:batch_end]
            batch_r = r[batch_begin:batch_end]
            

            yield len(batch_u),batch_u,batch_v,batch_r
                
            batch_begin = batch_end
            batch_end = batch_size+batch_begin
            if batch_end> num_pair:
                batch_end=num_pair
