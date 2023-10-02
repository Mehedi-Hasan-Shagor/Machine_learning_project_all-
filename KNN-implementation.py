#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dir1=r"D:\Knn.csv"
df=pd.read_csv(dir1)
print(df)
data=df.drop('User_id',axis=1)
print(data)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    data['Gender']=le.fit_transform(data['Gender'])
print(data)


# In[12]:


x=data[["Gender","Age","Estimate"]].values
y=data[["purchased"]].values
print("x: ",x)
print(y)


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train,"\n\n\n",x_test,"\n\n\n",y_test)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_test_pre=knn.predict(x_test)
y_train_pre=knn.predict(x_train)
print(y_test)


# In[34]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print("Accuracy of our data: ",accuracy_score(y_test,y_test_pre))


# In[35]:


predict_y=knn.predict([[0,1,8]])
print(predict_y)


# In[29]:


from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=1),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=1))


plt.contour(x1,x2,knn.predict(np.array([x1.ravel(),x1.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(x1.min(),x1.max())
plt.xlim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[:,0],x_set[:,1],c= ListedColormap(("red","green"))(i),
                 label=j)


# In[28]:


X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 
1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, 
step = 0.01))
plt.sactter(X1, X2, knn.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
         alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 
    'green'))(i), label = j)

plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[38]:


plt.scatter(x_train[y_train_pre==0,0],x_train[y_train_pre==0,1],s=100,c="blue",label="Cluster 1")
plt.scatter(x_train[y_train_pre==1,0],x_train[y_train_pre==1,1],s=100,c="red",label="Cluster 2")


# In[ ]:




