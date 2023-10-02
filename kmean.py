#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dir1=r"D:\k_mean.csv"
df=pd.read_csv(dir1)
df.head(50)


# In[7]:


x=df[['age','Annual_income','spending']].values
print(x)


# In[11]:


plt.scatter(x[:,0],x[:,1],c="black")
plt.xlabel("age")
plt.ylabel("Annual_income")
plt.show()


# In[18]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
kmeans.fit(x)


# In[19]:


kmeans.labels_


# In[20]:


kmeans.cluster_centers_


# In[21]:


y_predict=kmeans.predict(x)
plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c="blue",label="Cluster 1")
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c="red",label="Cluster 2")
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=100,c="green",label="Cluster 3")

centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],s=200,c="black",label="Center")
plt.legend()


# In[17]:


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


# In[ ]:




