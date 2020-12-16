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

v_most = {}

v_all_sum = 0
v_all_count = 0

def select_most(rs):
	r2c={}
	for r in rs:
		if not r in r2c:
			r2c[r]=0
		r2c[r]+=1

	max_r = 4
	max_c = 0

	for r in r2c:
		if r2c[r]>=max_c:
		 	max_r = r
		 	max_c = r2c[r]
	return max_r

for v in history_vr_lists:
	non_zero_list = []

	for r in history_vr_lists[v]:
		if r>0:
			non_zero_list.append(r)


	v_all_sum += sum(non_zero_list)
	v_all_count += len(non_zero_list)

	if len(non_zero_list)==0:
		v_most[v]=0
	else:
		v_most[v] = select_most(non_zero_list)



v_mean = 1.0*v_all_sum/v_all_count
print(v_mean)
v_mean =4

for v in history_vr_lists:
	if v_most[v]==0:
		v_most[v] = v_mean
res = []

for v in test_v:
	res.append(v_most[v])

print(res[:20])
print(test_r[:20])


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

rmse = sqrt(mean_squared_error(res,test_r))
mae = mean_absolute_error(res,test_r)

print('user ha',rmse,mae)

