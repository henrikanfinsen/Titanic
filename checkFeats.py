# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# %%
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5)) in [train['Survived'] == 1][feature].value_counts()


# %%
bar_chart('Embarked')

