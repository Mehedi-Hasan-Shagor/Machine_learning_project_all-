#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# In[2]:


file_dir=r"C:\Users\Random_Forest_Dataset.csv"


# In[3]:


df=pd.read_csv(file_dir)
df.head(150)


# In[31]:


from sklearn.preprocessing import LabelEncoder
labellencoder=LabelEncoder()
for i in df.columns:
    df[i]=labellencoder.fit_transform(df[i])
df.head(150)


# In[27]:


df.describe()


# In[4]:


x=df.drop('Species',axis=1)
y=df['Species']
print(x)


# In[7]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


# In[1]:


pip install scikit-learn


# In[3]:


pip install scikit-learn


# In[4]:


pip install --upgrade scikit-learn


# In[13]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
x


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.75,random_state=400)
print(x_train)


# In[17]:


sns.set_theme(style="darkgrid")
sns.countplot(y=y_train,data=df,palette="mako_r")
plt.ylabel('class')
plt.xlabel('Total')
plt.show()


# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[21]:


y_train_pre=rf.predict(x_train)
y_test_pre=rf.predict(x_test)

print(x_test[0])
print(y_test_pre[0])


# In[ ]:





# In[32]:


x_test_outside=[[-0.421,  -.005, 1.5645 ,1.5 , 0.02]]
y_pre=rf.predict(x_test_outside)
print(y_pre)


# In[27]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[29]:


print(confusion_matrix(y_train,y_train_pre))
print(confusion_matrix(y_test,y_test_pre))


# In[30]:


print(accuracy_score(y_train,y_train_pre))
print(accuracy_score(y_test,y_test_pre))


# In[31]:


print(classification_report(y_train,y_train_pre))
print(classification_report(y_test,y_test_pre))


# In[ ]:




