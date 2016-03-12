# Imports
from datetime import datetime
from datetime import timedelta
# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

train_user_df=pd.read_csv("tianchi_fresh_comp_train_user.csv")
train_item_df=pd.read_csv("tianchi_fresh_comp_train_item.csv")
print("--------------------user-info---------------")
train_user_df.info()
print("--------------------item-info---------------")
train_item_df.info()
train_user_df.drop(['user_geohash'],axis=1,inplace=True)
time_series=train_user_df['time']

train_user_df['time']=[datetime.strptime(x,"%Y-%m-%d %H") for x in time_series]
#real_time_series=Series(real_time)
#train_user_df=train_user_df.join(real_time_series)
#train_user_df=train_user_df.drop(['time'],axis=1,inplace=True)
train_user_df.info()
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=100, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=1,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )

for i_days in range(30):
	start_date=datetime(2014,12,18)

	vali_date=start_date-timedelta(i_days)

	train_vali_date=train_user_df[train_user_weekiter['time']==vali_date]
	
	#each training batch  (in a week)
	train_user_weekiter=train_user_df[(vali_date-train_user_weekiter['time']).days<7 and (vali_date-train_user_weekiter['time']).days >=0]

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

	for i_ter in train_user_weekiter.index:

		item=train_user_weekiter.ix['i_ter','item_category']

		action=train_user_weekiter.ix['i_ter','behavior_type']
		usr_id=train_user_weekiter.ix['i_ter','user_id']
		item_id=train_user_weekiter.ix['i_ter','item-id']

		train_vali_date.ix['']
		
		
		train_user_weekiter.ix['i_ter','behavior_type']

		view_num_i=np.shape(train_user_weekiter[train_user_weekiter['item_category']==item and train_user_weekiter['behavior_type']==1])[0]

		mark_num_i=np.shape(train_user_weekiter[train_user_weekiter['item_category']==item and train_user_weekiter['behavior_type']==2])[0]

		cart_num_i=np.shape(train_user_weekiter[train_user_weekiter['item_category']==item and train_user_weekiter['behavior_type']==3])[0]

		bought_num_i=np.shape(train_user_weekiter[train_user_weekiter['item_category']==item and train_user_weekiter['behavior_type']==4])[0]


		view_num.append(view_num_i)

		cart_num.append(cart_num_i)

		mark_num.append(mark_num_i)

		bought_num.append(bought_num_i)

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
		else action==4:
			view.append(0)
			mark.append(0)
			cart.append(0)
			bought.append(1)
    




