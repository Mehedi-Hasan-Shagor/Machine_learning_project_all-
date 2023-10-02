#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd


# In[30]:


c=r"C:\Users\k_mean.csv"


# In[31]:


df=pd.read_csv(c)



# In[32]:


x=df.iloc[:,[3,4]].values
#print(x)


# In[33]:


plt.scatter(x[:,0],x[:,1])


# In[12]:


from sklearn.cluster import KMeans
wcss_list=[]
for i in range(1,20):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
plt.plot(range(1,20),wcss_list)
plt.title("The Elbow Method Graph")
plt.xlabel("number of Clusters(k)")
plt.ylabel("Wcss_list")
plt.show()


# In[34]:


kmeans=KMeans(n_clusters=4,init="k-means++",random_state=42)
y_predict=kmeans.fit_predict(x)
print(y_predict)


# In[38]:


plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c="blue",label="Cluster 1")
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c="red",label="Cluster 2")
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=100,c="green",label="Cluster 3")
plt.scatter(x[y_predict==3,0],x[y_predict==3,1],s=100,c="orange",label="Cluster 4")
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],s=200,c="black",label="Center")
plt.legend()

