#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# DATA SCIENCE AND BUSINESS ANALYTICS TASKS
# 
# BY SHUBHA MINZ
# 
# TASK2: PREDICTIONUSING SUPERVISED ML

# In[1]:


#importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans


# In[2]:


os.chdir("D:/dataset")
df = pd.read_csv('Iris.csv')
df.head()


# Visualizing the data

# In[3]:


df.describe()


# In[4]:


plt.figure(figsize=(6, 6))
plt.scatter(df['SepalLengthCm'],df['SepalWidthCm'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


# In[5]:


plt.figure(figsize=(6, 6))
plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()


# Visualizing Clustering

# In[6]:


sns.FacetGrid(df, hue="Species", height=6)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()
plt.show()


# In[7]:


sns.FacetGrid(df, hue="Species", height=6)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend()
plt.show()


# Let's find the optimal number of cluster and apply K-Means Algorithm

# In[8]:


x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[9]:


plt.plot(range(1, 11), wcss,'*-')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.text(4,200000,"optimal number of clusters = 3")
plt.show()


# From the graph, it is clear that optimal number of clusters is 3
# 
# 
# 
# Let's apply K-Means to the dataset

# In[10]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# Let's visualise the cluster, but on first two columns only

# In[11]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')


# In[12]:


# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()


# THANK YOU!!

# In[ ]:




