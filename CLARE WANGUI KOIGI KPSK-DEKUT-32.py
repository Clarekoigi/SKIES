#Load libraries
import pandas as pd
import seaborn as sns

#i)Load data
wangui=pd.read_csv("Mall_Customers.csv")
print(wangui)

#ii)How many records are there?
print("Show number of records:")
print(wangui.shape)

#ii)What features is it composed of?
features=wangui.columns.tolist()
print("features:",features)

#iv)Does it contain missing values?
print("Show existence of missing values:")
print(wangui.isnull().sum())

#v)If missing values are present
#STEP 1:Remove rows
#STEP 2:Replace empty values with mean,mode,median etc

#vii)Relationship between some features

import matplotlib.pyplot as plt

#Plotting Gender vs. Annual income
plt.hist(wangui['Genre'],bins=5)
plt.xlabel('Gender')
plt.ylabel('Annual Income (k$)')
plt.title('Realationship between Gender and Annual Income(k$)')         
plt.show()         
#From the graph it is evident that Females earn more annual income than Males

#Plotting Gender vs. Age
plt.hist(wangui['Genre'],bins=5)
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Relationship between Gender and Age')
plt.show()
#From the graph it is evident that Females are more old than Males

#Plotting Gender vs.Spending Score
plt.hist(wangui['Genre'],bins=5)
plt.xlabel('Gender')
plt.ylabel('Spending Score (1-100)')
plt.title('Relationship between Gender and Spending Score (1-100)')
plt.show()
#From the graph it is evident that Females spend more than Males

#Using the kmeans method to place the data in groups(clusters) based on customer 'Age' and 'Annual Income'
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(wangui[['Age','Annual Income (k$)']],wangui[['Spending Score (1-100)']],test_size=0.33,random_state=0)
plt.show()

#normalizing
from sklearn import preprocessing
x_train_norm=preprocessing.normalize(x_train)
x_test_norm=preprocessing.normalize(x_test)

#fitting/training and evaluation
from sklearn.cluster import KMeans
exercise=KMeans(n_clusters=3,random_state=0,n_init='auto')
exercise.fit(x_train_norm)

#visualize the results
sns.scatterplot(data=x_train,x='Age',y='Annual Income (k$)',hue=exercise.labels_)
plt.show()

#look at the distribution of Spending Score in the 3 groups. A boxplot can be useful
sns.boxplot(x=exercise.labels_,y=y_train['Spending Score (1-100)'])
plt.show()

#people in the first and third cluster have similar distributions of Spending Score and are higher than that of the second cluster

#evaluating
#evaluate performance of the clustering alogarithm using a silhouette score which is part of sklearn.metrics .A lower score represents a better fit
from sklearn.metrics import silhouette_score
perf=silhouette_score(x_train_norm,exercise.labels_,metric='euclidean')
print(perf)

#how many clusters to use?
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
#visualize a few,start with K=2
sns.scatterplot(data=x_train,x='Age',y='Annual Income (k$)',hue=fits[0].labels_)
plt.show()

#2 halves,not good looking

#what about K
sns.scatterplot(data=x_train,x='Age',y='Annual Income (k$)',hue=fits[2].labels_)
plt.show()
#is it better?worse?
#what about 7
sns.scatterplot(data=x_train,x='Age',y='Annual Income (k$)',hue=fits[5].labels_)
plt.show()
#7 is so many
#use the elbow plot to compare.
sns.lineplot(x=K,y=score)
plt.show()
#choose the point where the performance start to flatten or get worse.Here K=5.
sns.scatterplot(data=x_train,x='Age',y='Annual Income (k$)',hue=fits[3].labels_)
plt.show()

#To attain the best number of groups (value of k),i used the elbow method which involves running the k-means alogarithm for a range of values of k.
#The 'elbow' point in the plot represents the optimal number of clusters.
#Once I've identified the optimal number of clusters,I run the k-means alogarithm with that number of clusters and visualized the resulting clusters.
#This way, I've used k-means method to place the data into groups based on customer age and annual income and determined the optimal number of clusters using the elbow method.

                


         


