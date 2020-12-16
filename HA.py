import os
import argparse
import sys
import pickle
from math import sqrt


parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
parser.add_argument('--dataset', type=str, default='toy_dataset', help='dataset')

args = parser.parse_args()


dir_data = './data/'+args.dataset

path_data = dir_data + ".pickle"
data_file = open(path_data, 'rb')
history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
    data_file)

test_num = len(test_u)//2
test_u = test_u[test_num:]
test_v = test_v[test_num:]
test_r = test_r[test_num:]

u_ha = {}
v_ha = {}

u_all_sum = 0
u_all_count = 0 


for u in history_ur_lists:
	# remove 0
	non_zero_list = []
	for r in history_ur_lists[u]:
		if r>0:
			non_zero_list.append(r)

	u_all_sum += sum(non_zero_list)
	u_all_count += len(non_zero_list)

	if len(non_zero_list)==0:
		u_ha[u]=0
	else:
	    u_ha[u] = sum(non_zero_list)*1.0/len(non_zero_list)


v_all_sum = 0
v_all_count = 0


for v in history_vr_lists:
	non_zero_list = []

	for r in history_vr_lists[v]:
		if r>0:
			non_zero_list.append(r)


	v_all_sum += sum(non_zero_list)
	v_all_count += len(non_zero_list)

	if len(non_zero_list)==0:
		v_ha[v]=0
	else:
		v_ha[v] = sum(non_zero_list)*1.0/len(non_zero_list)


v_mean = 1.0*v_all_sum/v_all_count
u_mean = 1.0*u_all_sum/u_all_count
print(u_mean,v_mean)


for v in history_vr_lists:
	if v_ha[v]==0:
		v_ha[v] = v_mean


for u in history_ur_lists:
	if u_ha[u]==0:
		u_ha[u]=u_mean




u_res = []
v_res = []
uv_res = []
for u,v in list(zip(test_u,test_v)):
    u_res.append(u_ha[u])
    v_res.append(v_ha[v])
    if u_ha[u]==u_mean:
    	u_res[-1]=v_res[-1]
    if v_ha[v]==v_mean:
    	v_res[-1]=u_res[-1]
    uv_res.append((u_res[-1]+v_res[-1])/2.0)



print(u_res[:7])
print(v_res[:7])
print(test_r[:7])


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

u_rmse = sqrt(mean_squared_error(u_res,test_r))
u_mae = mean_absolute_error(u_res,test_r)

v_rmse = sqrt(mean_squared_error(v_res,test_r))
v_mae = mean_absolute_error(v_res,test_r)

uv_rmse = sqrt(mean_squared_error(uv_res,test_r))
uv_mae = mean_absolute_error(uv_res,test_r)


print('user ha',u_rmse,u_mae)
print('item ha',v_rmse,v_mae)
print('uv ha',uv_rmse,uv_mae)

