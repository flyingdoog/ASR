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

uvr = {}
for u in history_u_lists:
	uvr[u]={}
	for v,r in list(zip(history_u_lists[u],history_ur_lists[u])):
		uvr[u][v]=r


v_ha = {}
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


for v in history_vr_lists:
	if v_ha[v]==0:
		v_ha[v] = v_mean

v_res = []
hit = 0
count = 0 
for u,v in list(zip(test_u,test_v)):
	if count%10000==0:
		print(count,hit)
	count+=1
	res = v_ha[v]

	social=[]
	nb_len = 1
	nbs = set()
	nbs.add(u)
	for hop in range(nb_len):
		next_nbs = nbs.copy()
		for root in nbs:
			for nb in social_adj_lists[root]:
				next_nbs.add(nb)
		nbs = next_nbs

	for nb in nbs:
		if v in uvr[nb]:
			social.append(uvr[nb][v])
	if len(social)>0:
		hit +=1
		res = sum(social)/len(social)
		print(v_ha[v],res)

	v_res.append(res)

print(v_res[:10])
print(test_r[:10])


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


v_rmse = sqrt(mean_squared_error(v_res,test_r))
v_mae = mean_absolute_error(v_res,test_r)

print('item ha',v_rmse,v_mae)

