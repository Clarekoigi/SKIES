import pandas as pd
wangui=pd.read_csv('housing.csv')
print(wangui.head())

#Visualize it
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,10))
sns.histplot(wangui['latitude'],kde=True)
plt.show()

sns.histplot(wangui['population'],kde=True)
plt.show()

                    
