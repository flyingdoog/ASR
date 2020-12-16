import os
import pickle
import time
from dataset import Data_loader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from utils import *
from math import sqrt
import argparse
import sys
from inits import *

user_filter = False

parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
parser.add_argument('--embed_dim', type=int, default=30, metavar='N', help='embedding size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--epochs', type=int, default=10000, metavar='N', help='number of epochs to train')
parser.add_argument('--device', type=int, default=0, help='device to use')
parser.add_argument('--dataset', type=str, default='ciao', help='dataset')
parser.add_argument('--batch_size', type=int, default=512, help='dataset')
parser.add_argument('--test_batch_size', type=int, default=-1, help='test batch size')
parser.add_argument('--model', type=str, default='DisGAT', help='dataset')
parser.add_argument('--socialmodel', type=str, default='GCN', help='dataset')
parser.add_argument('--print_every', type=int, default=20, help='print_every')
parser.add_argument('--n_heads', type=int, default=2, help='number of heads')
parser.add_argument('--decay', type=float, default=0.99, help='print_every')
parser.add_argument('--hiddens', type=str, default='128-32', help='dataset')
parser.add_argument('--gate_init_value', type=float, default=1, help='gate_init_value')
parser.add_argument('--dropout', type=float, default=1, help='dropout keep prob')
parser.add_argument('--lambda1', type = float, default=0, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--feature', action='store_true')
parser.add_argument('--len', type=int, default=2, help='len')
parser.add_argument('--early_stop', type=int, default=10, help='len')
parser.add_argument('--nuu', action='store_true')
parser.add_argument('--nuv', action='store_true')
parser.add_argument('--nvu', action='store_true')
parser.add_argument('--nuself', action='store_true')
parser.add_argument('--nvself', action='store_true')
parser.add_argument('--rgate', type=float, default=1, help='gate_init_value')
parser.add_argument('--mode', type=str, default='cat', help='gate_init_value')


modes = ['cat','sum','ave','max','FC']

att_dim = 15

args = parser.parse_args()
if args.mode not in modes:
    print('mode error',args.mode)
    quit()


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
import tensorflow as tf 
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)


global_step = tf.Variable(0, name='global_step', trainable=False)
feed_dict_val={}

from layers import Dense, GraphConvolution
from GAT import GAT
# from inits import *

best_rmse = 9999.0
best_mae = 9999.0
best_test_rmse = 9999.0
best_test_mae = 9999.0  

test_fil = []

def myfilter(history_u_lists, _min=0, _max=10):

    fil = []
    num = 0 
    num_user = len(history_u_lists)
    for uid in range(num_user):
        if len(history_u_lists[uid])> _min and len(history_u_lists[uid])<_max:
            fil.append(True)
            num += 1
        else:
            fil.append(False)
    return fil, num

class GraphRec():
    def __init__(self,num_users, num_items, num_ratings, embed_dim, data_loader,ratings_list):


        uu_adj = preprocess_adj(data_loader.uu_adj)
        uv_adj = preprocess_adj(data_loader.uv_adj)
        vu_adj = preprocess_adj(data_loader.vu_adj)


        uv_adjs = {}
        vu_adjs = {}
        for rating in ratings_list:
            uv_adjs[rating] = preprocess_adj(data_loader.uv_adjs[rating])
            vu_adjs[rating] = preprocess_adj(data_loader.vu_adjs[rating])


        self.learning_rate = args.lr
        learning_rate_frist = 1e-3
 
        self.u2e = glorot((num_users, embed_dim),name='user_embedding')
        self.v2e = glorot((num_items, embed_dim), name='item_embedding')


        vbais = glorot((num_items, 1), name='user_bias')
        ubais = glorot((num_items, 1), name='item_bias')

        self.uu_adj = tf.cast(tf.SparseTensor(uu_adj[0],uu_adj[1],uu_adj[2]),tf.float32)
        self.uv_adj = tf.cast(tf.SparseTensor(uv_adj[0],uv_adj[1],uv_adj[2]),tf.float32)
        self.vu_adj = tf.cast(tf.SparseTensor(vu_adj[0],vu_adj[1],vu_adj[2]),tf.float32)


        if args.model =='GCN':
            uu_adj_gcn = preprocess_adj(data_loader.uu_adj_gcn)
            # uvu_adj_gcn = preprocess_adj(data_loader.uvu_adj)
            # vuv_adj_gcn = preprocess_adj(data_loader.vuv_adj)
            self.uu_adj_gcn = tf.cast(tf.SparseTensor(uu_adj_gcn[0],uu_adj_gcn[1],uu_adj_gcn[2]),tf.float32)
            # self.uvu_adj_gcn = tf.cast(tf.SparseTensor(uvu_adj_gcn[0],uvu_adj_gcn[1],uvu_adj_gcn[2]),tf.float32)
            # self.vuv_adj_gcn = tf.cast(tf.SparseTensor(vuv_adj_gcn[0],vuv_adj_gcn[1],vuv_adj_gcn[2]),tf.float32)


        self.uv_adjs = {}
        self.vu_adjs = {}

        for rating in uv_adjs:
            self.uv_adjs[rating] = tf.cast(tf.SparseTensor(uv_adjs[rating][0],uv_adjs[rating][1],uv_adjs[rating][2]),tf.float32)
            self.vu_adjs[rating] = tf.cast(tf.SparseTensor(vu_adjs[rating][0],vu_adjs[rating][1],vu_adjs[rating][2]),tf.float32)

        self.embed_dim = embed_dim

        user_fea_size = self.u2e.get_shape().as_list()[1]
        item_fea_size = self.v2e.get_shape().as_list()[1]

        placeholders = {}
        # placeholders['dropout'] = args.dropout

        u_in = self.u2e
        v_in = self.v2e
        u_out = u_in
        v_out = v_in

        # if args.dropout<1:
        #     u_in = tf.nn.dropout(u_in,args.dropout)
        #     v_in = tf.nn.dropout(v_in,args.dropout)


        self.att_u = []
        self.att_v = []
        n_heads = args.n_heads

        u_dim = u_in.get_shape().as_list()[1]
        v_dim = v_in.get_shape().as_list()[1]

        if args.mode =='cat':
            DisGCN_len = u_dim//5
        else:
            DisGCN_len = u_dim
        
        act = tf.nn.relu

        u_in_extend = tf.expand_dims(u_in, 0)
        v_in_extend = tf.expand_dims(v_in, 0)

        for layer_id in range(args.len):
            print('number of layers',args.len)
            if args.socialmodel == 'GAT':
                uu_outs = []
                for head_index in range(n_heads):
                    uu_outs.append(GAT(adj_mat=self.uu_adj,output_dim=u_dim//n_heads,act=act)(u_in_extend,u_in_extend,num_users,num_users))
                uu_out = tf.concat(uu_outs,axis=-1)

            elif args.socialmodel == 'GCN':
                if args.model=='GCN':
                    placeholders['support'] = [self.uu_adj_gcn]
                else:
                    placeholders['support'] = [self.uu_adj]

                uu_out =GraphConvolution(input_dim = u_dim, output_dim = u_dim, placeholders = placeholders,act=act)(u_in)
            
            # u_in_extend = tf.expand_dims(u_in,0)
            # v_in_extend = tf.expand_dims(v_in,0)
            # uu_out = GAT(adj_mat=self.uu_adj,output_dim=u_dim)(u_in_extend,u_in_extend,num_users,num_users)
            
            uv_outs = []
            vu_outs = []

            models = ['DisGCN','DisGAT','GAT','GCN']
            if args.model not in models:
                print('model error')

            if args.model=='DisGCN' or args.model=='DisGAT':
                for rating  in self.vu_adjs:
                    if args.model=='DisGAT':
                        uv_out_temps = []
                        for head_index in range(n_heads):
                            uv_out_temps.append(GAT(adj_mat=self.vu_adjs[rating],output_dim=DisGCN_len//n_heads,act=act)(v_in_extend,u_in_extend,num_items,num_users))
                        uv_out_temp = tf.concat(uv_out_temps,axis=-1)
                        uv_outs.append(uv_out_temp)

                    elif args.model=='DisGCN':
                        placeholders['support'] = [self.vu_adjs[rating]]
                        uv_out_temp = GraphConvolution(input_dim = u_dim, output_dim = DisGCN_len, placeholders = placeholders,act=act)(u_in)
                        uv_outs.append(uv_out_temp)
            elif args.model=='GAT':
                uv_out_temps = []
                for head_index in range(n_heads):
                    uv_out_temps.append(GAT(adj_mat=self.vu_adj,output_dim=u_dim//n_heads,act=act)(v_in_extend,u_in_extend,num_items,num_users))
                uv_out_temp = tf.concat(uv_out_temps,axis=-1)
                uv_out_temp = tf.nn.dropout(uv_out_temp,args.dropout)
                uv_outs.append(uv_out_temp)

            elif args.model=='GCN':
                placeholders['support'] = [self.vu_adj]#[self.uvu_adj_gcn]#
                uv_outs.append(GraphConvolution(input_dim = u_dim, output_dim = u_dim, placeholders = placeholders,act=act)(u_in))
            else:
                print('unkonwn model')
                quit()

            if args.model=='DisGCN' or args.model=='DisGAT':
                for rating in self.uv_adjs:
                    if args.model == 'DisGAT':
                        vu_out_temps = []
                        for head_index in range(n_heads):
                            vu_out_temps.append(GAT(adj_mat=self.uv_adjs[rating],output_dim=DisGCN_len//n_heads,act=act)(u_in_extend,v_in_extend,num_users,num_items))

                        vu_outs.append(tf.concat(vu_out_temps,axis=-1))
                    elif args.model == 'DisGCN':
                        placeholders['support'] = [self.uv_adjs[rating]]
                        vu_out_temp = GraphConvolution(input_dim = v_dim, output_dim = DisGCN_len, placeholders = placeholders,act=act)(v_in)
                        vu_out_temp = tf.nn.dropout(vu_out_temp,args.dropout)
                        vu_outs.append(vu_out_temp)
            elif args.model == 'GAT':
                vu_out_temps = []
                for head_index in range(n_heads):
                    vu_out_temps.append(GAT(adj_mat=self.uv_adj,output_dim=v_dim//n_heads,act=act)(u_in_extend,v_in_extend,num_users,num_items))

                vu_outs.append(tf.concat(vu_out_temps,axis=-1))
            elif args.model == 'GCN':
                placeholders['support'] = [self.uv_adj]#[self.vuv_adj_gcn]#
                vu_outs.append(GraphConvolution(input_dim = v_dim, output_dim = v_dim, placeholders = placeholders,act=act)(v_in))
            else:
                print('unkonwn model')
                quit()


                   

            u_feas = []
            u_feas.append(u_in)


            vu_out_concat = tf.concat(vu_outs,axis=-1)
            uv_out_concat = tf.concat(uv_outs,axis=-1)


            # modes = ['cat','sum','ave','max','FC']

            if args.model=='GCN' or args.model=='GAT':
                vu_out = vu_outs[0]
                uv_out = uv_outs[0]

            elif args.mode =='cat':
                vu_out = vu_out_concat
                uv_out = uv_out_concat
            elif args.mode== 'FC':
                vu_out = Dense(input_dim = vu_out_concat.get_shape().as_list()[-1], output_dim = u_in.get_shape().as_list()[-1], activation = tf.nn.relu, name = 'uvout')(vu_out_concat)
                uv_out =  Dense(input_dim = uv_out_concat.get_shape().as_list()[-1], output_dim = v_in.get_shape().as_list()[-1], activation = tf.nn.relu, name = 'attention_network')(uv_out_concat)
            elif args.mode=='sum':
                vu_out = tf.add_n(vu_outs)
                uv_out = tf.add_n(uv_outs)
            elif args.mode=='ave':
                vu_out = tf.reduce_mean(vu_outs)
                uv_out = tf.reduce_mean(uv_outs)   
            elif args.mode=='max':
                vu_out = tf.reduce_max(vu_outs, reduction_indices=[0])
                uv_out = tf.reduce_max(uv_outs, reduction_indices=[0])
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


            # u_out = (1-gate_z1/2-gate_z2/2)*u_in_reset + (gate_z1/2)*vu_out+(gate_z2/2)*uu_out
            u_in_reset = gate_r*u_in

            u_out = u_in_reset + (gate_z1)*vu_out+(gate_z2)*uu_out


            v_feas = []
            v_feas.append(v_in)     
            v_feas.append(uv_out)

            # v_feas.extend(uv_outs)

            v_fea_concat = tf.concat(v_feas,axis=-1)
            if args.nvself:
                gate_vr = args.rgate
            else:
                gate_vr = Dense(input_dim = v_fea_concat.get_shape().as_list()[-1], output_dim = v_in.get_shape().as_list()[-1] , activation = tf.nn.sigmoid, bias = True, name = 'gate_r')(v_fea_concat)
            
            if args.nuv:
                gate_vz = args.gate_init_value
            else:
                gate_vz = Dense(input_dim = v_fea_concat.get_shape().as_list()[-1], output_dim = v_in.get_shape().as_list()[-1] , activation = tf.nn.sigmoid, bias = True, name = 'gate_z1')(v_fea_concat)

            # v_out = (1-gate_vz)*v_in_reset + gate_vz*uv_out
            v_out = gate_vr*v_in + gate_vz*uv_out
            
            # v_out = uv_out

            u_in = u_out
            v_in = v_out
            # if args.dropout<1:
            #     u_in = tf.nn.dropout(u_in,args.dropout)
            #     v_in = tf.nn.dropout(v_in,args.dropout)



        self.nodes_u    =  tf.placeholder(name='node_users',dtype=tf.int32,shape=[None])
        self.nodes_v    =  tf.placeholder(name='node_items',dtype=tf.int32,shape=[None])
        self.nodes_r    =  tf.placeholder(name='labels',dtype=tf.float32,shape=[None,1])


        x_u = tf.gather(u_out,self.nodes_u)
        x_v = tf.gather(v_out,self.nodes_v)

        b_u = tf.gather(ubais,self.nodes_u)
        b_v = tf.gather(vbais,self.nodes_v)


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

        self.outputs = self.w_uv_last(x)
        # self.outputs += b_u+b_v

        # self.labels = tf.squeeze(self.nodes_r)
        self.labels = self.nodes_r
        self.loss = tf.losses.mean_squared_error(self.labels,self.outputs)
        # self.loss = tf.losses.absolute_difference(self.labels,self.outputs)
        # self.loss = tf.reduce_mean((self.labels-self.outputs)*(self.labels-self.outputs))

        u_init = tf.gather(self.u2e,self.nodes_u)
        v_init = tf.gather(self.v2e,self.nodes_v)

        self.loss_emb_reg = tf.reduce_sum(tf.abs(u_init))+tf.reduce_sum(tf.abs(v_init))

        self.rmse = tf.sqrt(self.loss)
        self.loss += args.lambda1*self.loss_emb_reg

        learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,\
         global_step=global_step, decay_steps=1, decay_rate=args.decay, staircase=False)
        # learning_rate = self.learning_rate
        
        self.train_op=  tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.train_op_first=  tf.train.AdamOptimizer(learning_rate_frist).minimize(self.loss)
        
        # self.train_op=  tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        # self.train_op=  tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

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
        _, rmse, loss, output, att_u, att_v= sess.run([op,model.rmse, model.loss, model.outputs,model.att_u, model.att_v],feed_dict=feed_dict_val)
        t_end = time.time()
        t_train += t_end-t_begin
        output = np.squeeze(output)
        outputs.append(output)

        if count%args.print_every==0:
            # print('batch ',count, 'rmse',rmse)#,'t_load',t_load,'t_train',t_train)
            v_rmse, mae, loss = test(sess,model,data_loader,batch_size = args.test_batch_size, valid = True)
            # print('validation--------',v_rmse,mae)
            for i in range(len(att_u)):
                a_u = np.array(att_u[i][:10,:])
                a_v = np.array(att_v[i][:10,:])
                att_uv = np.hstack([a_u,a_v])
                print(np.array2string(att_uv, precision=2, separator=' ',suppress_small=True))
                print('--------')
            global best_mae
            global best_rmse
            global best_test_rmse
            global best_test_mae  
            if best_mae+best_rmse > mae+v_rmse:
                best_rmse = v_rmse
                best_mae = mae
                # saver = tf.train.Saver()
                # saver.save(sess, "./checkpoint/"+args.model)
                test_rmse, test_mae, test_loss = test(sess,model,data_loader,batch_size = args.test_batch_size, valid = False)
                best_test_rmse = test_rmse
                best_test_mae = test_mae
                print("test_loss %.4f v_rmse %.4f, mae:%.4f best: %.4f, best_mae: %.4f, test_rmse: %.4f, test_mae: %.4f" % (test_loss,v_rmse, mae,best_rmse, best_mae, test_rmse, test_mae))
            # print(output[:5])
            # print(batch_r[:5])

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

    # print(outputs[:5])
    # print(label[:5])
    outputs = np.clip(outputs,1,5)

    if user_filter:
        filter_out = []
        filter_label = []
        for user,out,l in list(zip(u,outputs,label)):
            if test_fil[user]:
                filter_out.append(out)
                filter_label.append(l)


        rmse = sqrt(mean_squared_error(filter_out, filter_label))
        mae = mean_absolute_error(filter_out, filter_label)

    else:
        rmse = sqrt(mean_squared_error(outputs, label))
        mae = mean_absolute_error(outputs, label)


    return rmse, mae, aloss


def main():


    embed_dim = args.embed_dim
    dir_data = './data/'+args.dataset

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(data_file)
    
    global test_fil

    test_fil,num = myfilter(history_u_lists)
    # print('number of testing samples ',num)
    ratings_list.pop(0)
    savePATH = 'models/'+args.model

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
    graphrec = GraphRec(num_users, num_items, num_ratings, embed_dim, data_loader,ratings_list)


    sess.run(tf.global_variables_initializer())


    endure_count = 0

  
    best_mae = 9999
    best_rmse = 9999
    v_rmse, mae,loss = test(sess,graphrec,data_loader,batch_size =args.test_batch_size)

    nmb_para = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('nmb_para',nmb_para)

    for epoch in range(1, args.epochs + 1):
        t_begin = time.time()
        train(sess,graphrec,data_loader,batch_size = args.batch_size, first=(epoch==1))
        t_end = time.time()
        print('s/epoch',t_end-t_begin)
        v_rmse, mae, loss  = test(sess,graphrec,data_loader,batch_size = args.test_batch_size, valid = True)


        if best_mae+best_rmse > mae+v_rmse:
            best_rmse = v_rmse
            best_mae = mae
            saver = tf.train.Saver()
            saver.save(sess, "./checkpoint/"+args.model)
            test_rmse, test_mae, test_loss = test(sess,graphrec,data_loader,batch_size = args.test_batch_size, valid = False)

            endure_count = 0
        else:
            endure_count += 1
        global best_test_rmse
        global best_test_mae
        print("v_rmse %.4f, mae:%.4f best: %.4f, best_mae: %.4f, test_rmse: %.4f, test_mae: %.4f best_test_rmse: %.4f, best_test_mae: %.4f" % (v_rmse, mae,best_rmse, best_mae, test_rmse, test_mae,best_test_rmse,best_test_mae))
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
