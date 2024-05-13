import pandas as pd
import seaborn as sns
df=pd.read_csv('retail_sales_dataset.csv')
print(df)

print("Show information of the dataset:\n")
print(df.info())

print("Show description of the dataset:\n")
print(df.describe)

print("Show existence of missing values:\n")
print(df.isnull().sum())

import matplotlib.pyplot as plt

#Plotting Gender vs. Product Category
plt.hist(df['Product Category'],bins=5)
plt.xlabel('Product Category')
plt.ylabel('Price per Unit')
plt.title('Relationship between Product Category and Price per Unit')
plt.show()

plt.figure(figsize=(8,10))
sns.boxplot(x="Gender",y="Age",data=df)
plt.show()

plt.figure(figsize=(8,10))
sns.boxplot(x="Product Category",y="Total Amount",data=df)
plt.show()

