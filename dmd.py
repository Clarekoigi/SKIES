import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('diamonds')
print(df.head())
plt.figure(figsize=(6,10))
sns.histplot(df['price'],kde=True)
plt.show()
