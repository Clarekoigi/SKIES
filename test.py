import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('diamonds')
print(df.head())
print(df.size)
print(df.shape)
print(df.info)
print(df.describe)



