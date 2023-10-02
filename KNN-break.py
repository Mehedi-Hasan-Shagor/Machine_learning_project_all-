#!/usr/bin/env python
# coding: utf-8

# In[22]:





# In[44]:


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
points={"blue":[[2,4],[1,3],[2,3],[3,2],[2,1],[5,7]],
       "red":[[5,6],[4,5],[4,6],[6,6],[5,4],[10,10]]}

new_point=[8,6]

def eucledian_distance(p,q):
    return np.sqrt(np.sum((np.array(p)-np.array(q))**2))
    
class KNN:
    def __int__(self, k=3):
        self.k=k
        self.point=none
    def fit(self,points):
        self.points=points
    def predict(self,new_point):
        distances=[]
        #print(distances)
        
        for category in self.points:
            #print(category)
            for point in self.points[category]:
                distance=eucledian_distance(point, new_point)
                #print(distance)
                distances.append([distance,category])
                #print(distances)
                
        categories=[category[1] for category in sorted(distances)]
       # print(categories)
        result=Counter(categories).most_common()[0][0]
        #print(result)
        return result

clf=KNN()
clf.fit(points)
print(clf.predict(new_point))


# In[45]:


#visualize
ax=plt.subplot()
ax.grid(True,color="black")
ax.figure.set_facecolor("White")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

for point in points['blue']:
    ax.scatter(point[0],point[1],color="#104DCA",s=60)
for point in points['red']:
    ax.scatter(point[0],point[1],color="#FF0000",s=60)

new_class=clf.predict(new_point)
color="#FF0000" if new_class == "red" else "#104DCA"
ax.scatter(new_point[0],new_point[1], color=color, marker="*",s=200, zorder=100)


# In[ ]:


from sklearn.model_selection import train_test_split

X


# In[ ]:




