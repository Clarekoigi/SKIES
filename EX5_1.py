import pandas as pd
wangui=pd.read_csv('housing.csv')
print(wangui.head())

#Visualize it
import matplotlib.pyplot as plt
import seaborn as sns
wangui=pd.read_csv('housing.csv',usecols=['longitude','latitude','median_house_value'])
sns.scatterplot(data=wangui,x='longitude',y='latitude',hue='median_house_value')
plt.show()

from sklearn.model_selection import train_test_split

#2.1
x_train,x_test,y_train,y_test=train_test_split(wangui[['latitude','longitude']],wangui[['median_house_value']],test_size=0.33,random_state=0)
plt.show()

#2.2 then normalize
from sklearn import preprocessing
x_train_norm=preprocessing.normalize(x_train)
x_test_norm=preprocessing.normalize(x_test)

#3 fitting/training and evaluation
from sklearn.cluster import KMeans
exercise=KMeans(n_clusters=3,random_state=0,n_init='auto')
exercise.fit(x_train_norm)

#then visualize the results
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=exercise.labels_)
plt.show()

#look at the distribution of median house prices in the 3 groups. A boxplot can be useful
sns.boxplot(x=exercise.labels_,y=y_train['median_house_value'])
plt.show()

#people in the first and third cluster have similar distributions of median house values and are higher than that of the second cluster

#3.2 evaluating
#evaluate performance of the clustering alogarithm using a silhouette score which is part of sklearn.metrics .A lower score represents a better fit
from sklearn.metrics import silhouette_score
perf=silhouette_score(x_train_norm,exercise.labels_,metric='euclidean')
print(perf)

#3.3 how many clusters to use?
#need to test a range of them
K=range(2,8)
fits=[]
score=[]
for k in K:
    #train the model for current value of K on training data
    model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(x_train_norm)
    #append the model to fits
    fits.append(model)
    #append the silhouette_score
    score.append(silhouette_score(x_train_norm,model.labels_,metric='euclidean'))
print(fits)
print(score)
#then visualize a few,start with K=2
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fits[0].labels_)
plt.show()
#2 halves,not good looking

#what about K
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fits[2].labels_)
plt.show()
#is it better?worse?
#what about 7
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fits[5].labels_)
plt.show()
#7 is so many
#use the elbow plot to compare.
sns.lineplot(x=K,y=score)
plt.show()
#choose the point where the performance start to flatten or get worse.Here K=5.
sns.scatterplot(data=x_train,x='longitude',y='latitude',hue=fits[3].labels_)
plt.show()

sns.boxplot(x=fits[3].labels_,y=y_train['median_house_value'])
plt.show()
                
             
                    
