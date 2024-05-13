import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
titanic_df = sns.load_dataset('titanic')
print(f"DataFrame shape : {titanic_df.shape}\n=================================")

print(f"DataFrame info : {titanic_df.info()}\n================================="
print(f"DataFrame columns : {titanic_df.columns}\n=================================")
print(f"The type of each column : {titanic_df.dtypes}\n=================================")
print(f"How much missing value in every column : {titanic_df.isna().sum()}\n=================================")
