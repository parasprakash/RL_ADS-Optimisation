
##### Importing Libraries#### 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### Importing Dataset ####

dataset0 = pd.read_csv('Historical Product Demand.csv',nrows=50000)
x=dataset0.iloc[:,0:2].values
y=dataset0.iloc[:,2].values
#y=np.asarray(y)
pd.unique(dataset0['Product_Category'])
pd.unique(dataset0['months'])
pd.unique(dataset0['Warehouse'])
from sklearn.preprocessing import OneHotEncoder,StandardScaler
encoder1= OneHotEncoder(categorical_features="all")
x=encoder1.fit_transform(x).toarray()
x=x[:,[1,2,4,5]]
y=y.reshape(-1,1)
ob1=StandardScaler()
x=ob1.fit_transform(x)
y=ob1.fit_transform(y)
x=np.append(arr=np.ones((50000,1)).astype(int),values=x,axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
y_pred=ob1.inverse_transform(y_pred)
y_test=ob1.inverse_transform(y_test)

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#### Data Preprocessing #### 

from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(missing_values=np.nan,strategy="constant",fill_value=0)
Imputer = Imputer.fit(dataset)
dataset = Imputer.transform(dataset)
dataset = pd.DataFrame(dataset)

#### Appling Random Reward Method ####

import random
ads_selected_random = []
total_reward_random =  0
for n in range(0, 10000):
    ad_random = random.randrange(10)
    ads_selected_random.append(ad_random)
    reward_random = dataset.values[n, ad_random]
    total_reward_random = total_reward_random + reward_random

#### Visualising The Results ####

plt.hist(ads_selected_random , width = 0.85 )
plt.title(' Random Reward')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.savefig("RandomReward.png")
plt.show()

#### Applying Upper Confidence Bound Method ####

import math
ads_selected_ucb = []
numbers_of_selections_ucb = [0] * 10
sums_of_rewards_ucb = [0] * 10
total_reward_ucb = 0
for n in range(0, 10000):
    ad_ucb = 0
    max_upper_bound_ucb = 0
    for i in range(0, 10):
        if (numbers_of_selections_ucb[i] > 0):
            average_reward_ucb = sums_of_rewards_ucb[i] / numbers_of_selections_ucb[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections_ucb[i])
            upper_bound_ucb = average_reward_ucb + delta_i
        else:
            upper_bound_ucb = 1e400
        if upper_bound_ucb > max_upper_bound_ucb:
            max_upper_bound_ucb = upper_bound_ucb
            ad_ucb = i
    ads_selected_ucb.append(ad_ucb)
    numbers_of_selections_ucb[ad_ucb] = numbers_of_selections_ucb[ad_ucb] + 1
    reward_ucb = dataset.values[n, ad_ucb]
    sums_of_rewards_ucb[ad_ucb] = sums_of_rewards_ucb[ad_ucb] + reward_ucb
    total_reward_ucb = total_reward_ucb + reward_ucb

#### Visualising the Results ####

plt.hist(ads_selected_ucb , width = 0.85)
plt.title(' Upper Confidence Bound')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.savefig('UCB.png')
plt.show() 


#### Implementing Thompson Sampling ####

ads_selected_thomson = []
numbers_of_rewards_1 = [0] * 10
numbers_of_rewards_0 = [0] * 10
total_reward_thomson = 0
for n in range(0, 10000):
    ad = 0
    max_random = 0
    for i in range(0, 10):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected_thomson.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward_thomson = total_reward_thomson + reward

#### Visualising the Results ####

plt.hist(ads_selected_thomson)
plt.title('Thomson Sampling')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.savefig('Thomson_Sampling.png')
plt.show()

#### END ####