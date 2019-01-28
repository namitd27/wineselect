#%%
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

#%%
wine = pd.read_csv('winequality.csv')

#%%
wine.sample(5)

#%%
wine.isnull().sum()

#%%
sns.heatmap(wine[wine.columns[:11]].corr(), annot=True)
fig = plt.gcf()
fig.set_size_inches(18,10)
plt.show()

#%%
sns.countplot(x='quality', data=wine)
plt.show()

#%%
fig = plt.figure()
wine.hist(grid=True, bins=15 )
fig = plt.gcf()
fig.set_size_inches(36,24)
plt.show()

#%%
def map_quality(row):
    if((row['quality'] > 0) & (row['quality'] < 4)):
        return 'Poor'
    if((row['quality'] >= 4) & (row['quality'] < 6)):
        return 'Average'
    if((row['quality'] >= 6) & (row['quality'] < 8)):
        return 'Above Average'
    if(row['quality'] >= 8):
        return 'Excellent'

#%%
wine['Rating'] = wine.apply(lambda row: map_quality(row), axis = 1)

#%%
sns.countplot(x='Rating', data=wine)
plt.show()

#%%
wine.to_csv('refinedWineQuality.csv')