# rank 220 in 2231 4.9
# Imports
from datetime import datetime
from datetime import timedelta
# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

train_user_df=pd.read_csv("tianchi_fresh_comp_train_user.csv")
train_item_df=pd.read_csv("tianchi_fresh_comp_train_item.csv")
print("--------------------user-info---------------")
train_user_df.info()
print("--------------------item-info---------------")
train_item_df.info()
train_user_df.drop(['user_geohash'],axis=1,inplace=True)
time_series=train_user_df['time']

print("-------------------translating_times-----------------")
train_user_df['time']=[datetime.strptime(x.split()[0],"%Y-%m-%d") for x in time_series]
#real_time_series=Series(real_time)
#train_user_df=train_user_df.join(real_time_series)
#train_user_df=train_user_df.drop(['time'],axis=1,inplace=True)
train_user_df.info()
#import lasagne
#from lasagne import layers
#from lasagne.updates import nesterov_momentum
#from nolearn.lasagne import NeuralNet
#from nolearn.lasagne import visualize

#net1 = NeuralNet(
        #layers=[('input', layers.InputLayer),
        #        ('hidden', layers.DenseLayer),
        #        ('output', layers.DenseLayer),
        #        ],
        # layer parameters:
        #input_shape=(None,8),
        #hidden_num_units=100, # number of units in 'hidden' layer
        #output_nonlinearity=lasagne.nonlinearities.softmax,
        #output_num_units=1,  # 1 target values for buy or not

        # optimization method:
        #update=nesterov_momentum,
        #update_learning_rate=0.0001,
        #update_momentum=0.9,

        #max_epochs=15,
        #verbose=1,
        #)
# setting the trainning_set
print ("-------building the network------")


net=buildNetwork(8, 20, 1,bias=True)
ds=SupervisedDataSet(8,1)


for i_days in range(0,31,7):

	start_date=datetime(2014,12,5)

	vali_date=start_date-timedelta(i_days)
	print("preprocessing the"+str(i_days)+"training data")
	train_vali_date=train_user_df[train_user_df['time']==vali_date]
	t1=datetime.now()
	#each training batch  (in a week)
	for weekiter in range(1,7):
		if weekiter==1:
			train_user_weekiter=train_user_df[train_user_df['time']==vali_date-timedelta(weekiter)]
		else :
			train_user_weekiter=pd.concat([train_user_weekiter,train_user_df[train_user_df['time']==vali_date-timedelta(weekiter)]])

    
	#item_cat=np.unique(train_user_weekiter['item_category'])
	#usr_cat=
	# for cat_inter,i in enumerate(item_cat):
	#	usr_item=train_user_weekiter[train_user_weekiter[ 'item_category'] ==cat_iter]
	#	
	#	view_num=usr_item[]	
	view_num=[]
	mark_num=[]
	cart_num=[]
	bought_num=[]

	view=[]
	mark=[]
	cart=[]	
	bought=[]
	label=[]
	item_dic={}
    
	for i_ter in train_user_weekiter.index:
		usr_id=train_user_weekiter.ix[i_ter,'user_id']		
		item_cat=train_user_weekiter.ix[i_ter,'item_category']
		action=train_user_weekiter.ix[i_ter,'behavior_type']
		if not item_dic.has_key((usr_id,item_cat)):
			item_dic[(usr_id,item_cat)]=[0,0,0,0]
		if action==1:
			item_dic[(usr_id,item_cat)][0]+=1
		elif action==2:
			item_dic[(usr_id,item_cat)][1]+=1
		elif action==3:
			item_dic[(usr_id,item_cat)][2]+=1
		else:
			item_dic[(usr_id,item_cat)][3]+=1
			
	for i_ter in train_user_weekiter.index:
		item=train_user_weekiter.ix[i_ter,'item_category']
		action=train_user_weekiter.ix[i_ter,'behavior_type']
		usr_id=train_user_weekiter.ix[i_ter,'user_id']
		item_id=train_user_weekiter.ix[i_ter,'item_id']

		if np.shape(train_vali_date[train_vali_date['item_id']==item_id][train_vali_date['user_id']==usr_id][train_vali_date['behavior_type']==4])[0]==0:
			label_i=0
		else:
			label_i=1

		view_num_i=item_dic[(usr_id,item)][0]
		mark_num_i=item_dic[(usr_id,item)][1]
		cart_num_i=item_dic[(usr_id,item)][2]
		bought_num_i=item_dic[(usr_id,item)][3]
		
		view_num.append(view_num_i)
		cart_num.append(cart_num_i)
		mark_num.append(mark_num_i)
		bought_num.append(bought_num_i)		
		label.append(label_i)

		if action==1:
			view.append(1)
			mark.append(0)
			cart.append(0)
			bought.append(0)
		elif action==2:
			view.append(1)
			mark.append(1)
			cart.append(0)
			bought.append(0)	
		elif action==3:
			view.append(1)
			mark.append(0)
			cart.append(1)
			bought.append(0)		
		else:
			view.append(0)
			mark.append(0)
			cart.append(0)
			bought.append(1)

    
	train_user_weekiter['cat_view']=np.array(view_num)
	train_user_weekiter['cat_cart']=np.array(cart_num)
	train_user_weekiter['cat_mark']=np.array(mark_num)
	train_user_weekiter['cat_bought']=np.array(view_num)
	
	train_user_weekiter['view_tag']=np.array(view)
	train_user_weekiter['mark_tag']=np.array(mark)
	train_user_weekiter['bought_tag']=np.array(bought)
	train_user_weekiter['cart_tag']=np.array(cart)
	train_user_weekiter['label_tag']=np.array(label)

	if i_days==0:
		training=train_user_weekiter
	else:
		training=pd.concat([training,train_user_weekiter])
	t2=datetime.now()
	print ("--------------processing time per week-------------")
	print t2-t1


#set the test_set
test_date=datetime(2014,12,19)

#vali_date=start_date-timedelta(i_days)

#train_vali_date=train_user_df[train_user_weekiter['time']==vali_date]

#each training batch  (in a week)
print ("preprocessing the testing_data")
for weekiter in range(1,7):
	if weekiter==1:
		test_user_weekiter=train_user_df[train_user_df['time']==test_date-timedelta(weekiter)]
	else :
		test_user_weekiter=pd.concat([test_user_weekiter,train_user_df[train_user_df['time']==test_date-timedelta(weekiter)]])
view_nu=[]
mark_nu=[]
cart_nu=[]
bought_nu=[]

view_t=[]
mark_t=[]
cart_t=[]	
bought_t=[]

	
item_dic1={}	
for i_ter in test_user_weekiter.index:
	usr_id=test_user_weekiter.ix[i_ter,'user_id']		
	item_cat=test_user_weekiter.ix[i_ter,'item_category']
	action=test_user_weekiter.ix[i_ter,'behavior_type']
	if not item_dic1.has_key((usr_id,item_cat)):
		item_dic1[(usr_id,item_cat)]=[0,0,0,0]
	if action==1:
		item_dic1[(usr_id,item_cat)][0]+=1
	elif action==2:
		item_dic1[(usr_id,item_cat)][1]+=1
	elif action==3:
		item_dic1[(usr_id,item_cat)][2]+=1
	else:
		item_dic1[(usr_id,item_cat)][3]+=1

for i_ter in test_user_weekiter.index:
	item_cat=test_user_weekiter.ix[i_ter,'item_category']
	action=test_user_weekiter.ix[i_ter,'behavior_type']
	usr_id=test_user_weekiter.ix[i_ter,'user_id']
	item_id=test_user_weekiter.ix[i_ter,'item_id']

	#if train_vali_date.ix[train_vali_date['item_id']==usr_id and train_vali_date['user_id']==usr_id and train_vali_date['behavior_type']==4]:
		#label_i=1
	#else:
		#label_i=0
	
	#t_action=test_user_weekiter.ix['i_ter','behavior_type']

	view_num_i=item_dic1[(usr_id,item_cat)][0]
	mark_num_i=item_dic1[(usr_id,item_cat)][1]
	cart_num_i=item_dic1[(usr_id,item_cat)][2]
	bought_num_i=item_dic1[(usr_id,item_cat)][3]


	view_nu.append(view_num_i)
	cart_nu.append(cart_num_i)
	mark_nu.append(mark_num_i)
	bought_nu.append(bought_num_i)		
	#label.append(label_i)

	if action==1:
		view_t.append(1)
		mark_t.append(0)
		cart_t.append(0)
		bought_t.append(0)
	elif action==2:
		view_t.append(1)
		mark_t.append(1)
		cart_t.append(0)
		bought_t.append(0)	
	elif action==3:
		view_t.append(1)
		mark_t.append(0)
		cart_t.append(1)
		bought_t.append(0)		
	else :
		view_t.append(0)
		mark_t.append(0)
		cart_t.append(0)
		bought_t.append(1)

test_user_weekiter['cat_view']=np.array(view_nu)
test_user_weekiter['cat_cart']=np.array(cart_nu)
test_user_weekiter['cat_mark']=np.array(mark_nu)
test_user_weekiter['cat_bought']=np.array(bought_nu)

test_user_weekiter['view_tag']=np.array(view_t)
test_user_weekiter['mark_tag']=np.array(mark_t)
test_user_weekiter['bought_tag']=np.array(bought_t)
test_user_weekiter['cart_tag']=np.array(cart_t)
#test_user_weekiter['label_tag']=np.array(label_t)


#net1.fit(training[['cat_view','cat_cart','cat_mark','cat_bought','view_tag','mark_tag','bought_tag','cart_tag']],training[['label_tag']])
ds.setField('input',training[['cat_view','cat_cart','cat_mark','cat_bought','view_tag','mark_tag','bought_tag','cart_tag']])
ds.setField('target',training[['label_tag']])
print ("----------------------make test data----------------")
out=SupervisedDataSet(8,1)

test_item_pre=pd.merge(train_item_df,test_user_weekiter,on=['item_id','item_category'],how='inner')
#out.setField('input',test_item_pre[['cat_view','cat_cart','cat_mark','cat_bought','view_tag','mark_tag','bought_tag','cart_tag']])
#out.setField('target',np.zeros(np.shape(test_item_pre)[0],1))
for i_ter in test_item_pre.index:
	out.addSample(test_item_pre.ix[i_ter,['cat_view','cat_cart','cat_mark','cat_bought','view_tag','mark_tag','bought_tag','cart_tag']],0)


print ("---------------------------predict_length----------------------")
print np.shape(test_item_pre)[0]

print ("----------------------start training----------------")
trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence(maxEpochs=5)


#test_user_weekiter.index=test_user_weekiter['item_id']
#train_item_df.index=train_item_df['item_id']

print ("----------------------predict----------------")
pre=net.activateOnDataset(out)

#BUY_predict=net1.predict(test_item_pre)
test_item_pre['predict']=pre


test_item_pre=test_item_pre.sort_index(by=['predict'],ascending=False)[0:7000]
print ("----------------------write to file----------------")
submission = pd.DataFrame({
        "user_id": test_item_pre['user_id'],
        "item_id": test_item_pre['item_id']
    })
submission.to_csv('tianchi_mobile_recommendation_predict.csv', index=False)
see_pre = pd.DataFrame({
        "pre": test_item_pre['predict']
    })
see_pre.to_csv('see_pre.csv', index=False)

