#!/usr/bin/env python
# coding: utf-8

# In[56]:


import matplotlib.pyplot as plt

x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12,15,20,17,23,11,20]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21,5,14,10,11,14,23]
classes = [0, 1, 1, 0, 0, 1, 1, 0, 1, 1,1,0,1,1,0,0]

plt.scatter(x, y, c=classes)
plt.show() 


# In[57]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np

data = list(zip(x, y))
#X=data
#data
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(data, classes) 
new_x =15
new_y = 10
new_point =np.array([(new_x, new_y)])
#new_point=np.array([new_x,new_y])

prediction = knn.predict(new_point)

#print(prediction)
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
new_point
distances = np.linalg.norm(data - new_point, axis=1)
distances


# In[79]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(data,classes,test_size=0.2,shuffle=True) #random_state=1


# In[80]:


y_test


# In[81]:


classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)


# In[82]:


prediction=classifier.predict(X_test)
prediction


# In[83]:


from sklearn.metrics import accuracy_score, classification_report
print("Accuracy",accuracy_score(y_test,prediction))


# In[84]:


print("Classification report")
print(classification_report(y_test,prediction))


# In[73]:


import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.scatter(X_train, X_train, marker= '*',edgecolors='black')
plt.title("Predicted values with k=5", fontsize=20)


# In[85]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,prediction)
sns.heatmap(cm,annot=True,fmt="d")


# In[ ]:




