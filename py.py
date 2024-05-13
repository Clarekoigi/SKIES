import plotly.express as px
import seaborn as sns

#x = [1,2,3,4]
#y = [1,2,3,4]
df = sns.load_dataset('diamonds')

#fig = px.histogram(df,x='cut')
#fig = px.violin(df,x='cut',y='carat')
fig = px.scatter(df,x='cut',y='price')
fig.show()

