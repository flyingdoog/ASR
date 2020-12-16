import os
import pickle
import numpy as np
import time
from datasetgcn import Data_loader
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from global_parameters import *
from utils import *
from math import sqrt
import argparse
import sys

class our_GCN():
    def __init__(self,num_users, num_items, num_ratings, embed_dim, data_loader,ratings_list):


        uu_adj = preprocess_adj(data_loader.uu_adj)
        uv_adj = preprocess_adj(data_loader.uv_adj)
        vu_adj = preprocess_adj(data_loader.vu_adj)


        uv_adjs = {}
        vu_adjs = {}

        self.learning_rate = args.lr
        learning_rate_frist = 1e-3
 
        self.u2e = glorot((num_users, embed_dim),name='user_embedding')
        self.v2e = glorot((num_items, embed_dim), name='item_embedding')

        self.uu_adj = tf.cast(tf.SparseTensor(uu_adj[0],uu_adj[1],uu_adj[2]),tf.float32)
        self.uv_adj = tf.cast(tf.SparseTensor(uv_adj[0],uv_adj[1],uv_adj[2]),tf.float32)
        self.vu_adj = tf.cast(tf.SparseTensor(vu_adj[0],vu_adj[1],vu_adj[2]),tf.float32)


        if args.model =='GCN':
            uu_adj_gcn = preprocess_adj(data_loader.uu_adj_gcn)
            uvu_adj_gcn = preprocess_adj(data_loader.uvu_adj)
            vuv_adj_gcn = preprocess_adj(data_loader.vuv_adj)
            self.uu_adj_gcn = tf.cast(tf.SparseTensor(uu_adj_gcn[0],uu_adj_gcn[1],uu_adj_gcn[2]),tf.float32)
            self.uvu_adj_gcn = tf.cast(tf.SparseTensor(uvu_adj_gcn[0],uvu_adj_gcn[1],uvu_adj_gcn[2]),tf.float32)
            self.vuv_adj_gcn = tf.cast(tf.SparseTensor(vuv_adj_gcn[0],vuv_adj_gcn[1],vuv_adj_gcn[2]),tf.float32)


        self.embed_dim = embed_dim

        placeholders = {}

        u_in = self.u2e
        v_in = self.v2e
        u_out = u_in
        v_out = v_in


        self.att_u = []
        self.att_v = []
        n_heads = args.n_heads

        u_dim = u_in.get_shape().as_list()[1]
        v_dim = v_in.get_shape().as_list()[1]        
        act = tf.nn.relu


        for layer_id in range(args.len):
            print('number of layers',args.len)
            placeholders['support'] = [self.uu_adj_gcn]
            uu_out =GraphConvolution(input_dim = u_dim, output_dim = u_dim, placeholders = placeholders,act=act)(u_in)
            
            placeholders['support'] = [self.uvu_adj_gcn]#[self.vu_adj]
            vu_out = (GraphConvolution(input_dim = u_dim, output_dim = u_dim, placeholders = placeholders,act=act)(u_in))
            placeholders['support'] = [self.vuv_adj_gcn]#[self.uv_adj]
            uv_out = (GraphConvolution(input_dim = v_dim, output_dim = v_dim, placeholders = placeholders,act=act)(v_in))
        
            u_feas = []
            u_feas.append(u_in)
            u_feas.append(vu_out)
            u_feas.append(uu_out)
            u_fea_concat = tf.concat(u_feas,axis=-1)
           
            if args.nuself:
                gate_r = args.rgate
            else:
                gate_r  = Dense(input_dim = u_fea_concat.get_shape().as_list()[-1], output_dim = u_in.get_shape().as_list()[-1] , activation = tf.nn.sigmoid, bias = True, name = 'gate_r')(u_fea_concat)

            if args.nuu:
                gate_z2 = args.gate_init_value
            else:
                gate_z2 = Dense(input_dim = u_fea_concat.get_shape().as_list()[-1], output_dim = u_in.get_shape().as_list()[-1] , activation = tf.nn.sigmoid, bias = True, name = 'gate_z2')(u_fea_concat)
            
            if args.nvu:
                gate_z1 =args.gate_init_value
            else:
                gate_z1 = Dense(input_dim = u_fea_concat.get_shape().as_list()[-1], output_dim = u_in.get_shape().as_list()[-1] , activation = tf.nn.sigmoid, bias = True, name = 'gate_z1')(u_fea_concat)

            u_in_reset = gate_r*u_in

            u_out = u_in_reset + (gate_z1)*vu_out+(gate_z2)*uu_out


            v_feas = []
            v_feas.append(v_in)     
            v_feas.append(uv_out)

            v_fea_concat = tf.concat(v_feas,axis=-1)
            if args.nvself:
                gate_vr = args.rgate
            else:
                gate_vr = Dense(input_dim = v_fea_concat.get_shape().as_list()[-1], output_dim = v_in.get_shape().as_list()[-1] , activation = tf.nn.sigmoid, bias = True, name = 'gate_r')(v_fea_concat)
            
            if args.nuv:
                gate_vz = args.gate_init_value
            else:
                gate_vz = Dense(input_dim = v_fea_concat.get_shape().as_list()[-1], output_dim = v_in.get_shape().as_list()[-1] , activation = tf.nn.sigmoid, bias = True, name = 'gate_z1')(v_fea_concat)

            v_out = gate_vr*v_in + gate_vz*uv_out
            
            u_in = u_out
            v_in = v_out


        self.nodes_u    =  tf.placeholder(name='node_users',dtype=tf.int32,shape=[None])
        self.nodes_v    =  tf.placeholder(name='node_items',dtype=tf.int32,shape=[None])
        self.nodes_r    =  tf.placeholder(name='labels',dtype=tf.float32,shape=[None,1])


        x_u = tf.gather(u_out,self.nodes_u)
        x_v = tf.gather(v_out,self.nodes_v)

        x_uv = tf.concat([x_u, x_v], axis=-1)
        x_uv_dim = x_uv.get_shape().as_list()[1]

        hidden_layer_dims = [64,16]
        if args.hiddens!='':
            hidden_layer_dims = []
            ss = args.hiddens.split('-')
            for s in ss:
                if int(s)!=0:
                    hidden_layer_dims.append(int(s))
            print(hidden_layer_dims)

        self.w_uvs = []

        if len(hidden_layer_dims)==0:
            self.w_uv_last = Dense(input_dim = x_uv_dim,output_dim = 1, name = 'graphrec_w_uv_last')
        else:
            self.w_uvs.append(Dense(input_dim = x_uv_dim, output_dim= hidden_layer_dims[0], name = 'graphrec_w_uv0',activation= tf.nn.relu))
            
            i = 0
            for i in range(1,len(hidden_layer_dims)):
                self.w_uvs.append(Dense(input_dim = hidden_layer_dims[i-1],output_dim = hidden_layer_dims[i], name = 'graphrec_w_uv'+str(i), activation= tf.nn.relu))


            self.w_uv_last = Dense(input_dim = hidden_layer_dims[-1],output_dim = 1, name = 'graphrec_w_uv_last')

        
        x = x_uv
        for layer in self.w_uvs:
            x = tf.layers.batch_normalization(x)
            x = layer(x)
            # x = tf.layers.batch_normalization(x,momentum=0.5)
            # x = tf.nn.dropout(x,args.dropout)



        self.outputs = self.w_uv_last(x)
        self.labels = self.nodes_r
        self.loss = tf.losses.mean_squared_error(self.labels,self.outputs)

        u_init = tf.gather(self.u2e,self.nodes_u)
        v_init = tf.gather(self.v2e,self.nodes_v)

        self.loss_emb_reg = tf.reduce_sum(tf.abs(u_init))+tf.reduce_sum(tf.abs(v_init))

        self.rmse = tf.sqrt(self.loss)
        self.loss += args.lambda1*self.loss_emb_reg

        learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,\
         global_step=global_step, decay_steps=1, decay_rate=args.decay, staircase=False)
        
        self.train_op=  tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.train_op_first=  tf.train.AdamOptimizer(learning_rate_frist).minimize(self.loss)
        