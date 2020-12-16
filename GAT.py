import numpy as np
import tensorflow as tf
from layers import *


class GAT(object):
    def __init__(self, adj_mat, output_dim,act=tf.nn.relu):
        self.adj_mat = adj_mat
        self.output_dim = output_dim
        self.act = act



    def __call__(self, u_inputs,v_inputs,u_size,v_size):
        
        x = v_inputs
        adj_mat = self.adj_mat

        
        # simplest self-attention possible
        f_1 = tf.layers.conv1d(u_inputs, 1, 1)
        f_2 = tf.layers.conv1d(v_inputs, 1, 1)
        
        f_1 = tf.reshape(f_1, (u_size, 1))
        f_2 = tf.reshape(f_2, (v_size, 1))

        seq_fts = tf.layers.conv1d(x, self.output_dim, 1, use_bias=False)


        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        coefs = tf.sparse_reshape(coefs, [u_size, v_size])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        print('--------vals.shape------',vals.shape)
        # vals = tf.expand_dims(vals, axis=0)
        # vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)
        return self.act(ret)  # activation
