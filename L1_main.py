import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib.pyplot import figure


data_path = os.path.join(str(Path.home()), 'PycharmProjects/ML/src/adult/')

df = pd.read_csv(os.path.join(data_path, "adult_b.data"))

# Назначаем имена колонок от поставщика данных
columns = ('age workclass fnlwgt education educ-num marital-status occupation relationship '
           'race sex capital-gain capital-loss  hours-per-week native-country salary')

df.columns = columns.split()  #этот метод разделит датасет по колонкам как в массиве columns
print(df.info())

print(df.head())
df.drop_duplicates()

print("len: ", len(df))

print(len(df[df['age'] == 79])) # условие для вывода данных

print("\nunic in workclass: ", df['workclass'].value_counts(), '\n')  # вернет количество уникальных совпадений в колонке. возможны фильтры

print(df == ' ?')

f, a = plt.subplots()
plt.show()


fig = plt.figure(figsize = (10, 15))
sns.heatmap(df == ' ?', cbar=False)

fig.show()
