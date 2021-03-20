#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# DATA SCIENCE AND BUSINESS ANALYTICS TASKS
# 
# (BY SHUBHA MINZ)
# 
# TASK1: PREDICTIONUSING SUPERVISED ML
# 
# PREDICT THE PERCENTAGE OF A STUDENT BASED ON NO.OF STUDY OF HOURS

# In[24]:


#importing all the useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# importing dataset
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data=pd.read_csv(url)
print("Data successfully imported!")
data.head()


# In[3]:


#Getting basic information of dataset
data.info()


# In[4]:


#getting detailed information about dataset
data.describe()


# In[5]:


#selecting Independent and Dependent variable from Dataset
x= data.iloc[: , :1]
#x
y= data.iloc[:,1:]
#y


# In[6]:


#how much data is realated to respective coloumn
data.corr()


# In[7]:


#Plotting the distribution of scores
data.plot (x='Hours',y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# LINEAR REGRESSION

# In[8]:


from sklearn.metrics import make_scorer, SCORERS
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regression = LinearRegression()
mse = cross_val_score(lin_regression , x , y, scoring = 'neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# RIDGE REGRESSION

# In[9]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters = {'alpha':[1e-15 ,1e-10 ,1e-8, 1e-3 , 1e-2, 1, 5, 10, 20, 30 ,35, 40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge , parameters , scoring = 'neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)


# In[10]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# LASSO REGRESSION

# In[11]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
parameters = {'alpha':[1e-15 ,1e-10 ,1e-8, 1e-3 , 1e-2, 1, 5, 10, 20, 30 ,35, 40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso , parameters , scoring = 'neg_mean_squared_error',cv=5)
lasso_regressor.fit(x,y)


# In[12]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[13]:


from sklearn.model_selection import train_test_split
x_train , x_test ,y_train ,y_test = train_test_split(x,y,test_size =0.3 ,random_state=0)


# In[14]:


prediction_lasso = lasso_regressor.predict(x_test)
print(prediction_lasso)
prediction_ridge = ridge_regressor.predict(x_test)
print(prediction_ridge)


# In[16]:


import seaborn as sns
lassopredictions = y_test-prediction_lasso.reshape(8,1)
sns.displot(lassopredictions)


# In[17]:


import seaborn as sns
sns.displot(y_test-prediction_ridge)


# FROM THE ABOVE TWO DISPLOT GRAPH WE CAME TO KNOW THAT RIDGE AND LASSO REGRESSION ARE ALMOST SAME.

# In[18]:


#testing data-In Hours
print(x_test)
#Predicting the scores
y_pred = ridge_regressor.predict(x_test)


# In[19]:


df = pd.DataFrame({'Actual':[y_test], 'Predicted':[y_pred]})
df


# HERE FROM ABOVE THREE MODLES I AM SELECTING RIDGE REGRESSION BECAUSE LASSO IS USED WHEN THERE ARE MORE NO OF DATA AND HERE WE HAVE LIMITED DATA.

# In[23]:


#YOU CAN TEST YOUR OWN DATA
hours=9.25
own_pred = ridge_regressor.predict([[hours]])
print("No. of Hours={}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

