import os
os.system('sh /home/work/lixingjian/low_level_tasks/kill.sh')

import pickle
import numpy as np
import time
from dataset import Data_loader
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from global_parameters import *
from utils import *
from math import sqrt
import argparse
import sys
# max_num_user = 10

parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--device', type=int, default=0, help='device to use')
parser.add_argument('--dataset', type=str, default='toy_dataset', help='dataset')
parser.add_argument('--batch_size', type=int, default=512, help='dataset')
parser.add_argument('--test_batch_size', type=int, default=-1, help='test batch size')
parser.add_argument('--print_every', type=int, default=100, help='print_every')
parser.add_argument('--decay', type=float, default=0.99, help='print_every')

parser.add_argument('--dropout', type=float, default=1, help='dropout keep prob')
parser.add_argument('--lambda1', type = float, default=0, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stop', type=int, default=3, help='len')




args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
import tensorflow as tf 
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)

from layers import Dense, GraphConvolution
from inits import *

global_step = tf.Variable(0, name='global_step', trainable=False)

feed_dict_val={}


class MLPCF():

    def __init__(self,num_users, num_items, embed_dim):

        self.learning_rate = args.lr
        learning_rate_frist = 1e-3

        ufea = np.load('./data/'+args.dataset+'/ufea.npy')
        vfea = np.load('./data/'+args.dataset+'/vfea.npy')
        self.ufea = tf.Variable(ufea,name='user_feature',dtype=tf.float32)
        self.vfea = tf.Variable(vfea,name='item_feature',dtype=tf.float32)

        self.u2e = glorot((num_users, embed_dim),name='user_embedding')
        self.v2e = glorot((num_items, embed_dim), name='item_embedding')

        self.u2e = tf.concat([self.ufea,self.u2e],axis=-1)
        self.v2e = tf.concat([self.vfea,self.v2e],axis=-1)

        self.embed_dim = embed_dim

        placeholders = {}
        placeholders['dropout'] = args.dropout

        self.nodes_u    =  tf.placeholder(name='node_users',dtype=tf.int32,shape=[None])
        self.nodes_v    =  tf.placeholder(name='node_items',dtype=tf.int32,shape=[None])
        self.nodes_r    =  tf.placeholder(name='labels',dtype=tf.float32,shape=[None,1])

        x_u = tf.gather(self.u2e,self.nodes_u)
        x_v = tf.gather(self.v2e,self.nodes_v)

        x_uv = tf.concat([x_u, x_v], axis=-1)
        # x_uv_dim = self.embed_dim*2 + ufea.shape[1]+vfea.shape[1]

        x_uv_dim =x_uv.get_shape().as_list()[1]

        print(x_uv_dim)

        hidden_layer_dims = [self.embed_dim,64,32]
        self.w_uvs = []
        self.w_uvs.append(Dense(input_dim = x_uv_dim, output_dim= hidden_layer_dims[0], name = 'graphrec_w_uv0',activation= tf.nn.relu))
        
        i = 0
        for i in range(1,len(hidden_layer_dims)):
            self.w_uvs.append(Dense(input_dim = hidden_layer_dims[i-1],output_dim = hidden_layer_dims[i], name = 'graphrec_w_uv'+str(i), activation= tf.nn.leaky_relu))


        self.w_uv_last = Dense(input_dim = hidden_layer_dims[-1],output_dim = 1, name = 'graphrec_w_uv'+str(i))
        
        x = x_uv
        for layer in self.w_uvs:
            x = tf.layers.batch_normalization(layer(x),momentum=0.5)
            x = tf.nn.dropout(x,args.dropout)


        self.outputs = self.w_uv_last(x)

        self.labels = self.nodes_r
        self.loss = tf.losses.mean_squared_error(self.labels,self.outputs)

        loss_emb_reg = tf.reduce_sum(tf.abs(x_u))\
                    +tf.reduce_sum(tf.abs(x_v))

        self.rmse = tf.sqrt(self.loss)
        self.loss += args.lambda1*loss_emb_reg

        learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,\
         global_step=global_step, decay_steps=1, decay_rate=args.decay, staircase=False)

        
        self.train_op=  tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.train_op_first=  tf.train.AdamOptimizer(learning_rate_frist).minimize(self.loss)
        
def train(sess, model, data_loader, batch_size = 1024, first=False):
 
    outputs = []
    count = 0
    t_train =0
    t_end = time.time()
    t_begin = time.time()
    t_load  =0
    for real_batch_size, batch_u,batch_v,batch_r in data_loader.get_batch(train=True,batch_size = batch_size):
        feed_dict_val.update({model.nodes_u: batch_u})
        feed_dict_val.update({model.nodes_v: batch_v})
        re_batch_r = np.reshape(batch_r, (real_batch_size, 1)) 
        feed_dict_val.update({model.nodes_r: re_batch_r})
        # print(batch_r)
        # feed_dict_val.update({model.batch_size: real_batch_size})
        
        t_begin = time.time()
        t_load += t_begin-t_end
        op = model.train_op_first
        if not first:
            op = model.train_op
        _, rmse, loss, output= sess.run([op,model.rmse, model.loss, model.outputs],feed_dict=feed_dict_val)
        t_end = time.time()
        t_train += t_end-t_begin
        output = np.squeeze(output)
        outputs.append(output)
        if count%args.print_every==0:
            print('batch ',count, 'rmse',rmse)#,'t_load',t_load,'t_train',t_train)
            
            print(output[:5])
            print(batch_r[:5])

        count+=1
    
    outputs = np.concatenate(outputs,axis=-1)
    # return outputs

def test(sess, model, data_loader,batch_size=1024, valid = True):
    

    if batch_size ==-1:
        if valid:
            num, u,v,r = data_loader.get_valid()
        else:
            num, u,v,r = data_loader.get_valid_test()
        label = r

        feed_dict_val.update({model.nodes_u: u})
        feed_dict_val.update({model.nodes_v: v})
        r = np.reshape(r, (num, 1)) 
        feed_dict_val.update({model.nodes_r: r})
        outputs,aloss,rmse = sess.run([model.outputs,model.loss,model.rmse],feed_dict=feed_dict_val)
        outputs = np.squeeze(outputs)
    else:
        outputs = []
        losses = []
        for real_batch_size,batch_u,batch_v,batch_r in data_loader.get_batch(train = False,batch_size = batch_size):
            feed_dict_val.update({model.nodes_u: batch_u})
            feed_dict_val.update({model.nodes_v: batch_v})
            batch_r = np.reshape(batch_r, (real_batch_size, 1)) 
            feed_dict_val.update({model.nodes_r: batch_r})

            # feed_dict_val.update({model.batch_size: real_batch_size})

        
            output,loss = sess.run([model.outputs,model.loss],feed_dict=feed_dict_val)
            outputs.append(np.squeeze(output))
            losses.append(loss)

        outputs = np.concatenate(outputs,axis=-1)
        aloss = sum(losses)/len(losses)
   
        label = data_loader.test_label

    print(outputs[:5])
    print(label[:5])
    outputs = np.clip(outputs,1,5)
    expected_rmse = sqrt(mean_squared_error(outputs, label))
    mae = mean_absolute_error(outputs, label)
    return expected_rmse, mae, aloss, rmse


def main():

    embed_dim = args.embed_dim
    dir_data = './data/'+args.dataset

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)

    ratings_list.pop(0)

    # savePATH = 'models/'+args.model

    num_users = history_u_lists.__len__()
    print(num_users)
    num_items = history_v_lists.__len__()
    print(num_items)
    num_ratings = ratings_list.__len__()
    print(num_ratings)
    sys.stdout.flush()

    data_loader = Data_loader(train_u,train_v,train_r,\
        test_u,test_v,test_r,\
        history_u_lists, history_ur_lists, history_v_lists,history_vr_lists, social_adj_lists,ratings_list)

    # model
    mlpcf = MLPCF(num_users, num_items, embed_dim)


    sess.run(tf.global_variables_initializer())

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    best_test_rmse = 9999.0
    best_test_mae = 9999.0    

    expected_rmse, mae,loss,v_rmse = test(sess,mlpcf,data_loader,batch_size =args.test_batch_size)

    for epoch in range(1, args.epochs + 1):
        train(sess,mlpcf,data_loader,batch_size = args.batch_size, first=(epoch==1))
        expected_rmse, mae, loss, v_rmse = test(sess,mlpcf,data_loader,batch_size = args.test_batch_size, valid = True)

        if best_mae > mae:
            best_rmse = expected_rmse
            best_mae = mae
            saver = tf.train.Saver()
            saver.save(sess, "./checkpoint/MLPCF")
            test_rmse, test_mae, test_loss,test_rmse = test(sess,mlpcf,data_loader,batch_size = args.test_batch_size, valid = False)

            endure_count = 0
        else:
            endure_count += 1
        print("v_rmse %.4f, mae:%.4f best: %.4f, best_mae: %.4f, test_rmse: %.4f, test_mae: %.4f" % (v_rmse, mae,best_rmse, best_mae, test_rmse, test_mae))
        if endure_count > args.early_stop:
            break
    
    with open('./tuning.log','a') as fout:
    
        fout.write('embed_dim: '+str(args.embed_dim)+'\n')
        fout.write('lr: '+str(args.lr)+'\n')
        fout.write('dataset: '+str(args.dataset)+'\n')
        fout.write('batch_size: '+str(args.batch_size)+'\n') 
        fout.write('dropout: '+str(args.dropout)+'\n')
        fout.write('lambda1: '+str(args.lambda1)+'\n')
        fout.write('best_rmse: '+str(best_rmse)+'\n')
        fout.write('best_mae: '+str(best_mae)+'\n\n') 
        

if __name__ == "__main__":
    main()
