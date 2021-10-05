# Підготовка до роботи та ознайомлення з набором 

Загружаю всі необхідні для роботи пакети:


```python
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats.kde import gaussian_kde

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
```

Завантажую файл з даними в ноутбук:


```python
df = pd.read_csv('project.xls')
```

Дивлюся інформацію по наявному датафрейму:


```python
df.shape
```




    (32561, 15)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32561 entries, 0 to 32560
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   age             32561 non-null  int64 
     1   workclass       32561 non-null  object
     2   fnlwgt          32561 non-null  int64 
     3   education       32561 non-null  object
     4   education.num   32561 non-null  int64 
     5   marital.status  32561 non-null  object
     6   occupation      32561 non-null  object
     7   relationship    32561 non-null  object
     8   race            32561 non-null  object
     9   sex             32561 non-null  object
     10  capital.gain    32561 non-null  int64 
     11  capital.loss    32561 non-null  int64 
     12  hours.per.week  32561 non-null  int64 
     13  native.country  32561 non-null  object
     14  income          32561 non-null  object
    dtypes: int64(6), object(9)
    memory usage: 3.7+ MB
    

Бачу, що набір даних містить інформацію про 32561 об'єкт по 15 різним показникам без нявних пропусків.

Далі дивлюся перших 5 записів, щоб отримати уявлення про представлення даних.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>?</td>
      <td>77053</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82</td>
      <td>Private</td>
      <td>132870</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Widowed</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>18</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>?</td>
      <td>186061</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Widowed</td>
      <td>?</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>4356</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>Private</td>
      <td>140359</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Divorced</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Private</td>
      <td>264663</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Separated</td>
      <td>Prof-specialty</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3900</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



Виділяю дані про людей, які заробляють більше 50К доларів.


```python
df[df['income'] != '<=50K'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>74</td>
      <td>State-gov</td>
      <td>88638</td>
      <td>Doctorate</td>
      <td>16</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Other-relative</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>3683</td>
      <td>20</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41</td>
      <td>Private</td>
      <td>70037</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>Craft-repair</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>3004</td>
      <td>60</td>
      <td>?</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>10</th>
      <td>45</td>
      <td>Private</td>
      <td>172274</td>
      <td>Doctorate</td>
      <td>16</td>
      <td>Divorced</td>
      <td>Prof-specialty</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>3004</td>
      <td>35</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>11</th>
      <td>38</td>
      <td>Self-emp-not-inc</td>
      <td>164526</td>
      <td>Prof-school</td>
      <td>15</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>2824</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>12</th>
      <td>52</td>
      <td>Private</td>
      <td>129177</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Widowed</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>2824</td>
      <td>20</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
  </tbody>
</table>
</div>



Змінюю представлення даних в колонці "income" для полегшення роботи.


```python
dict_income = {label: idx for idx, label in enumerate(np.unique(df['income']))}
df['income'] = df['income'].map(dict_income)
df['income'].head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: income, dtype: int64



Змінивши на більш зручне представлення заробітку подивимося на відсоткове відношення в даних.


```python
print(f'''Кількість людей із заробітком менше 50К $: {np.round((df['income'].value_counts()[0]) / df.shape[0], 2) * 100}%.
Кількість людей із заробітком менше 50К $: {np.round((df['income'].value_counts()[1]) / df.shape[0], 2) * 100}%.''')
```

    Кількість людей із заробітком менше 50К $: 76.0%.
    Кількість людей із заробітком менше 50К $: 24.0%.
    

Далі необхідно попрацювати з пропущеними даними. У загальній інформації про датафрейм не було вказано, що є пропущені значення, однак можна замітити, що тут пропуски зображуються знаками питання (?), тому знаходимо скільки даних пропусків в кожному стовпці.


```python
columns_list = df.columns
pass_amount = {}

for column in columns_list:
    pass_in_column = df[df[column] == '?'][column].count()
    pass_amount[column] = pass_in_column

print(pass_amount)
```

    {'age': 0, 'workclass': 1836, 'fnlwgt': 0, 'education': 0, 'education.num': 0, 'marital.status': 0, 'occupation': 1843, 'relationship': 0, 'race': 0, 'sex': 0, 'capital.gain': 0, 'capital.loss': 0, 'hours.per.week': 0, 'native.country': 583, 'income': 0}
    

Видаляємо строки, де відсутня інформація в стовпцях.


```python
df_cleaned = df[(df['workclass'] != '?') & 
                (df['occupation'] != '?') & 
                (df['native.country'] != '?')]
```


```python
df_cleaned.shape
```




    (30162, 15)




```python
diff = df.shape[0] - df_cleaned.shape[0]

print(f"Кількість об'єктів зменшується на {diff}.\n\
Що становить {np.round(((diff / df.shape[0]) * 100), 2)}% від усіх заданих об'єктів.")
```

    Кількість об'єктів зменшується на 2399.
    Що становить 7.37% від усіх заданих об'єктів.
    

Бачимо, що набір зменшився на 2399 записів, що становить 7.37% усього набору.
Подивимось, яке співвідношення людей по двом цільовим показникам в очищеному наборі.


```python
print(f'''Кількість людей із заробітком менше 50К $: {np.round((df_cleaned['income'].value_counts()[0]) / df_cleaned.shape[0], 2) * 100}%.
Кількість людей із заробітком менше 50К $: {np.round((df_cleaned['income'].value_counts()[1]) / df_cleaned.shape[0], 2) * 100}%.''')
```

    Кількість людей із заробітком менше 50К $: 75.0%.
    Кількість людей із заробітком менше 50К $: 25.0%.
    

Співвідношення після очищення суттєво незмінилося, тому можемо працювати з очищеними набором. 

Наступним є необхідність перевірки на дублікати. Для цього використовуємо надані пакетом pandas методи:


```python
df_cleaned.duplicated().sum()
```




    23



Бачимо, що є дані, які дублюються. Тому необхідно очищити набір від них. Знову же використовуємо для цього наявний арсенал пандасу:


```python
df_cleaned = df_cleaned.drop_duplicates(keep="first")
```


```python
df_cleaned.duplicated().sum()
```




    0



Дивимося загальний опис набору, щоб побачити розподіл даних у кожному стовпці:


```python
df_cleaned.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education.num</th>
      <th>marital.status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital.gain</th>
      <th>capital.loss</th>
      <th>hours.per.week</th>
      <th>native.country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30139.000000</td>
      <td>30139</td>
      <td>3.013900e+04</td>
      <td>30139</td>
      <td>30139.000000</td>
      <td>30139</td>
      <td>30139</td>
      <td>30139</td>
      <td>30139</td>
      <td>30139</td>
      <td>30139.000000</td>
      <td>30139.000000</td>
      <td>30139.000000</td>
      <td>30139</td>
      <td>30139.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>16</td>
      <td>NaN</td>
      <td>7</td>
      <td>14</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>Private</td>
      <td>NaN</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United-States</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>22264</td>
      <td>NaN</td>
      <td>9834</td>
      <td>NaN</td>
      <td>14059</td>
      <td>4034</td>
      <td>12457</td>
      <td>25912</td>
      <td>20366</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27487</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.441720</td>
      <td>NaN</td>
      <td>1.897950e+05</td>
      <td>NaN</td>
      <td>10.122532</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1092.841202</td>
      <td>88.439928</td>
      <td>40.934703</td>
      <td>NaN</td>
      <td>0.249046</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.131426</td>
      <td>NaN</td>
      <td>1.056586e+05</td>
      <td>NaN</td>
      <td>2.548738</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7409.110596</td>
      <td>404.445239</td>
      <td>11.978753</td>
      <td>NaN</td>
      <td>0.432468</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>NaN</td>
      <td>1.376900e+04</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>NaN</td>
      <td>1.176275e+05</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>NaN</td>
      <td>1.784170e+05</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>47.000000</td>
      <td>NaN</td>
      <td>2.376045e+05</td>
      <td>NaN</td>
      <td>13.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>NaN</td>
      <td>1.484705e+06</td>
      <td>NaN</td>
      <td>16.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Зараз можемо побачити, що в деяких числох даних наявні викиди (наприклад в capital.gain). Тому необхідно візуалізувати дані та починати обробляти. Розпочну з років.

# Аналіз набору по стовпцям

Розпочинаємо з "age":


```python
df_cleaned['age'].hist(bins=20, density=True, alpha=0.7)
```




    <AxesSubplot:>




    
![png](output_36_1.png)
    



```python
def kde_image(column_name):
    density = gaussian_kde(df_cleaned[column_name])
    x = np.linspace(df_cleaned[column_name].min(), df_cleaned[column_name].max(), 1000)

    _ = plt.plot(x, density(x), color='green')
    plt.show()
    
def double_kde_image(column_name):
    density0 = gaussian_kde(df_cleaned[df_cleaned['income'] == 0][column_name])
    density1 = gaussian_kde(df_cleaned[df_cleaned['income'] == 1][column_name])
    
    minimum = min(df_cleaned[df_cleaned['income'] == 0][column_name].min(),
                 df_cleaned[df_cleaned['income'] == 1][column_name].min())
    maximum = max(df_cleaned[df_cleaned['income'] == 0][column_name].max(),
                 df_cleaned[df_cleaned['income'] == 1][column_name].max())
    
    x = np.linspace(minimum, maximum, 1000)

    _ = plt.plot(x, density0(x), color='red')
    _ = plt.plot(x, density1(x), color='blue')
    plt.show()
```


```python
kde_image('age')
```


    
![png](output_38_0.png)
    


Візуально розприділення по рокам не схоже на нормальне. Спробуємо поділити залежно від цільового показника.


```python
def hist_image(column_name, bins):
    colors = ['red', 'blue']

    for k, v in dict_income.items():
        _ = plt.hist(df_cleaned[df_cleaned['income'] == v][column_name], bins=bins, density=True, alpha=0.6, label=k)
        plt.axvline(df_cleaned[df_cleaned['income'] == v][column_name].mean(), alpha = 0.6, linestyle='dashed', color = colors[v])
        plt.legend()
        plt.xlabel(column_name)
        plt.ylabel('amount')

    plt.show()
```


```python
hist_image('age', 15)
```


    
![png](output_41_0.png)
    



```python
double_kde_image('age')
```


    
![png](output_42_0.png)
    


Спробуємо подивитися через box plot:


```python
ax = df_cleaned.boxplot(column='age', by='income')
_ = ax.get_figure().suptitle('')
```


    
![png](output_44_0.png)
    


Перевіримо чи однакові розподіли мають дві вибірки. Для цього використаємо критерій Манна-Уітні.


```python
def mann_whitneyu(column_name):
    x0 = df_cleaned[df_cleaned['income'] == 0].groupby(column_name)['income'].count().values
    x1 = df_cleaned[df_cleaned['income'] == 1].groupby(column_name)['income'].count().values

    res_mann_wh = stats.mannwhitneyu(x0, x1)
    print(f'p-value: {res_mann_wh[1]}.')
```


```python
mann_whitneyu('age')
```

    p-value: 4.966815651573901e-06.
    

По значенню p-value відхиляємо гіпотезу про подібність двох розподілів та говоримо, що вибірки мають різні розподіли.

Перевіримо гіпотезу про середнє значення обох вибірок:


```python
def ranksums(column_name):
    x0 = df_cleaned[df_cleaned['income'] == 0].groupby(column_name)['income'].count().values
    x1 = df_cleaned[df_cleaned['income'] == 1].groupby(column_name)['income'].count().values

    res_ranksum = stats.ranksums(x0, x1)
    print(f'p-value: {res_ranksum[1]}.')
```


```python
ranksums('age')
```

    p-value: 9.850793807452426e-06.
    

Результат свідчить про статистично значиму різницю середніх значень.

Далі переходимо до стовпця "workclass":


```python
def barh_image(column_name, fig_x, fig_y):
    fig, ax = plt.subplots(2, 1, figsize=(fig_x, fig_y))

    y0 = df_cleaned[df_cleaned['income'] == 0][column_name].value_counts().index
    y1 = df_cleaned[df_cleaned['income'] == 1][column_name].value_counts().index
    
    w0 = (df_cleaned[df_cleaned['income'] == 0][column_name].value_counts().values / df_cleaned[df_cleaned['income'] == 0].shape[0]) * 100
    w1 = (df_cleaned[df_cleaned['income'] == 1][column_name].value_counts().values / df_cleaned[df_cleaned['income'] == 1].shape[0]) * 100
    
    ax[0].set(title='"<=50K"')
    ax[0].barh(y=y0, width=w0)
    ax[0].set_xlabel('percent')
    
    ax[1].set(title='">50K"')
    ax[1].barh(y=y1, width=w1)
    ax[1].set_xlabel('percent')
    
    plt.tight_layout()
    plt.show()
```


```python
barh_image('workclass', 8, 5)
```


    
![png](output_55_0.png)
    


Спробуємо подивитися все на одному графіку:


```python
def bar_image(column_name, max_val, fig_x, fig_y):
    labels = (df_cleaned[df_cleaned['income'] == 0][column_name].value_counts().sort_index().index | df_cleaned[df_cleaned['income'] == 1][column_name].value_counts().sort_index().index).to_list()
    
    labels_0 = df_cleaned[df_cleaned['income'] == 0][column_name].value_counts().sort_index().index
    labels_1 = df_cleaned[df_cleaned['income'] == 1][column_name].value_counts().sort_index().index
    
    pre_x0 = (df_cleaned[df_cleaned['income'] == 0][column_name].value_counts().sort_index().values / df_cleaned[df_cleaned['income'] == 0].shape[0]) * 100
    pre_x1 = (df_cleaned[df_cleaned['income'] == 1][column_name].value_counts().sort_index().values / df_cleaned[df_cleaned['income'] == 1].shape[0]) * 100
    
    x0 = []
    x1 = []
    
    n_x0 = 0
    n_x1 = 0
    
    for label in labels:
        if label in labels_0:
            x0.append(pre_x0[n_x0])
            n_x0 += 1
        else:
            x0.append(0)
        
        if label in labels_1:
            x1.append(pre_x1[n_x1])
            n_x1 += 1
        else:
            x1.append(0)
    
    x = np.arange(len(labels))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(fig_x, fig_y))

    rect1 = ax.bar(x - width/2, x0, width, label='"<=50K"')
    rect2 = ax.bar(x + width/2, x1, width, label='">50K"')
    
    for i in range(len(x)):
        plt.text(i, x0[i], np.round(x0[i], 2), ha = 'right', fontsize=fig_y)
        plt.text(i, x1[i], np.round(x1[i], 2), ha = 'left', fontsize=fig_y)
    
    ax.set_title(column_name.title())
    ax.set_ylabel('percent')
    
    plt.xticks(ticks = x, labels = labels, rotation = 'horizontal')
    plt.yticks([n for n in range(0, max_val, 1)])
    plt.legend()
    
    fig.tight_layout()
    
    plt.show()
```


```python
bar_image('workclass', 78, 20, 20)
```


    
![png](output_58_0.png)
    



```python
x0 = df_cleaned[df_cleaned['income'] == 0]['workclass'].value_counts().index
x1 = df_cleaned[df_cleaned['income'] == 1]['workclass'].value_counts().index

inter = x0 & x1
not_inter = [wcl for wcl in x0 if wcl not in x1]

print(f'Категорії, які є в обох вибірках: {inter.to_list()}.\n')

print(f'Категорія, яка є тільки у вибірці 0: {not_inter}.')
```

    Категорії, які є в обох вибірках: ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc'].
    
    Категорія, яка є тільки у вибірці 0: ['Without-pay'].
    

Спостерігається різниця між двома цільовими класами у всіх робочих класах.

Далі переходимо до "fnlwgt":


```python
hist_image('fnlwgt', 50)
```


    
![png](output_62_0.png)
    



```python
double_kde_image('fnlwgt')
```


    
![png](output_63_0.png)
    


По гістограмі можна відмітити, що наявна асиметричність розподілу для обох ознак. Тому можемо використати W-критерій Вілкоксона для порівняння двох вибірок.


```python
x0 = df_cleaned[df_cleaned['income'] == 0]['fnlwgt']
x1 = df_cleaned[df_cleaned['income'] == 1]['fnlwgt']

res = stats.ranksums(x0, x1)
print(f'p-value: {res[1]}.')
```

    p-value: 0.07996067175376413.
    

p-value > 0.05. Дві вибірки мають однаковий розподіл та статистично невідрізняються. Можна зробити висновок, що даний показник не впливатиме на прогнозування і його можна відкинути.
Також це підверджує інтуїтивне відчуття, щодо цього стовпця.


```python
df_cleaned = df_cleaned.drop(columns='fnlwgt')
```

Переходимо до "education":


```python
bar_image('education', 40, 20, 10)
```


    
![png](output_69_0.png)
    


По гістограмі видно, що деякі види освіти зустрічаються частіше в одному класі відносно іншого. Спробую їх об'єднати в окремий клас:


```python
np.sort(df_cleaned['education'].unique())
```




    array(['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
           'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
           'Masters', 'Preschool', 'Prof-school', 'Some-college'],
          dtype=object)




```python
lt0 = ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Preschool']
lt1 = ['Prof-school', 'Doctorate']

def func(x):
    if x in lt0:
        return 'School'
    elif x in lt1:
        return 'Step'
    else:
        return x
        
df_cleaned['education'] = df_cleaned['education'].apply(func)
df_cleaned['education'].unique()
```




    array(['HS-grad', 'School', 'Some-college', 'Step', 'Bachelors',
           'Masters', 'Assoc-voc', 'Assoc-acdm'], dtype=object)




```python
df_cleaned['education']
```




    1             HS-grad
    3              School
    4        Some-college
    5             HS-grad
    6              School
                 ...     
    32556    Some-college
    32557      Assoc-acdm
    32558         HS-grad
    32559         HS-grad
    32560         HS-grad
    Name: education, Length: 30139, dtype: object



Також спостерігаємо значну різницю між рівнями освіти. Це може бути потрібним при прогнозуванні.


```python
df_cleaned['education'].unique()
```




    array(['HS-grad', 'School', 'Some-college', 'Step', 'Bachelors',
           'Masters', 'Assoc-voc', 'Assoc-acdm'], dtype=object)



Переходимо до "education.num":


```python
hist_image('education.num', 16)
```


    
![png](output_77_0.png)
    



```python
lt = df_cleaned['education.num'].unique().tolist()
print(sorted(lt))
```

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    


```python
double_kde_image('education.num')
```


    
![png](output_79_0.png)
    


Зробимо інтервальну шкалу кількості освіт для зменшення дисперсії:


```python
dict_edu_num = {(1, 2, 3): 1, (4, 5, 6): 2, (7, 8, 9, 10): 3, (11, 12, 13): 4, (14, 15, 16): 5}

def func(x):
    for k in dict_edu_num:
        if x in k:
            return dict_edu_num[k]

df_cleaned['education.num'] = df_cleaned['education.num'].apply(func)
df_cleaned['education.num'].head()
```




    1    3
    3    2
    4    3
    5    3
    6    2
    Name: education.num, dtype: int64



Спробуємо подивитися провірити гіпотезу про однаковість розподілу:


```python
hist_image('education.num', 3)
```


    
![png](output_83_0.png)
    



```python
mann_whitneyu('education.num')
```

    p-value: 0.20169765244631416.
    

Можна побачити, що розподіл відрізняється. Тому показник можна використовувати для класифікації.

Перевіряємо гіпотезу на рівність середніх значень:


```python
ranksums('education.num')
```

    p-value: 0.34720763934942456.
    

Результат показує користь нульової гіпотези. Можна вважати середні значення рівними.

Добавлю новий парамет для покращення диференціювання класів.


```python
df_cleaned['education.num'].unique()
```




    array([3, 2, 5, 4, 1], dtype=int64)




```python
df_cleaned['lvl_edu'] = df_cleaned['education.num'] / df_cleaned['age']
```


```python
hist_image('lvl_edu', 10)
```


    
![png](output_92_0.png)
    


Цей показник свідчить про кількість освіт відносно віку (чим молодший об'єкт та чим більше має освіт, тим значення вище).

Заміжній статус:


```python
bar_image('marital.status', 86, 20, 10)
```


    
![png](output_95_0.png)
    


Різницю можна замітити (особливо виділяється "marries-civ-spouse")

Місце роботи:


```python
bar_image('occupation', 30, 25, 10)
```


    
![png](output_98_0.png)
    


З графіку видно, що різниця є.

"relationship":


```python
bar_image('relationship', 80, 25, 15)
```


    
![png](output_101_0.png)
    



```python
np.sort(df_cleaned['relationship'].unique())
```




    array(['Husband', 'Not-in-family', 'Other-relative', 'Own-child',
           'Unmarried', 'Wife'], dtype=object)




```python
lt0 = ['Other-relative', 'Own-child', 'Unmarried']

def func(x):
    if x in lt0:
        return 'Relative-child'
    else:
        return x
        
df_cleaned['relationship'] = df_cleaned['relationship'].apply(func)
df_cleaned['relationship'].unique()
```




    array(['Not-in-family', 'Relative-child', 'Husband', 'Wife'], dtype=object)



"race":


```python
bar_image('race', 95, 20, 15)
```


    
![png](output_105_0.png)
    


"sex":


```python
bar_image('sex', 90, 5, 10)
```


    
![png](output_107_0.png)
    


Різниця в статі простежується.

"caputal.gain":


```python
hist_image('capital.gain', 100)
```


    
![png](output_110_0.png)
    



```python
double_kde_image('capital.gain')
```


    
![png](output_111_0.png)
    


p-value > 0.05, що може свідчити про прийняття H0 про відсутність різниці.


```python
df_cleaned['capital.gain'].describe()
```




    count    30139.000000
    mean      1092.841202
    std       7409.110596
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max      99999.000000
    Name: capital.gain, dtype: float64




```python
df_cleaned[df_cleaned['income'] == 0]['capital.gain'].describe()
```




    count    22633.000000
    mean       149.031989
    std        936.815624
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max      41310.000000
    Name: capital.gain, dtype: float64




```python
df_cleaned[df_cleaned['income'] == 0]['capital.gain'].value_counts().sort_index()
```




    0        21690
    114          6
    401          1
    594         28
    914          8
             ...  
    7978         1
    10566        6
    22040        1
    34095        3
    41310        2
    Name: capital.gain, Length: 88, dtype: int64




```python
df_cleaned[df_cleaned['income'] == 0]['capital.gain'].value_counts().sort_index()[-6:-1]
```




    7443     5
    7978     1
    10566    6
    22040    1
    34095    3
    Name: capital.gain, dtype: int64




```python
for i in df_cleaned[df_cleaned['income'] == 0]['capital.gain'].value_counts().sort_index()[-6:-1].index:
    df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned['capital.gain'] == i].index)
```


```python
df_cleaned[df_cleaned['income'] == 1]['capital.gain'].describe()
```




    count     7506.000000
    mean      3938.729017
    std      14387.833124
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max      99999.000000
    Name: capital.gain, dtype: float64




```python
df_cleaned[df_cleaned['income'] == 1]['capital.gain'].value_counts().sort_index()
```




    0        5911
    3103       88
    4386       56
    4687        3
    4787       22
    4934        7
    5178       91
    5556        4
    6097        1
    6418        7
    6514        4
    7298      240
    7430        8
    7688      270
    7896        2
    8614       52
    9386       16
    9562        4
    10520      43
    10605       9
    11678       2
    13550      25
    14084      39
    14344      26
    15020       5
    15024     337
    15831       6
    18481       2
    20051      33
    25124       2
    25236      11
    27828      32
    99999     148
    Name: capital.gain, dtype: int64



Перевіримо гіпотезу про однаковість закону розподілу:


```python
mann_whitneyu('capital.gain')
```

    p-value: 0.004549419623097.
    

Можна побачити, що розподіл відрізняється. 

Середні значення двох вибірок відрізняються.

"capital.loss":


```python
hist_image('capital.loss', 5)
```


    
![png](output_125_0.png)
    



```python
df_cleaned[df_cleaned['income'] == 0]['capital.loss'].describe()
```




    count    22617.000000
    mean        53.535438
    std        310.516424
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max       4356.000000
    Name: capital.loss, dtype: float64




```python
df_cleaned[df_cleaned['income'] == 0]['capital.loss'].value_counts().sort_index()[-8:-1]
```




    2457    1
    2467    1
    2603    4
    2754    2
    3683    1
    3770    2
    3900    2
    Name: capital.loss, dtype: int64




```python
df_cleaned[df_cleaned['income'] == 1]['capital.loss'].describe()
```




    count    7506.000000
    mean      193.802292
    std       592.896137
    min         0.000000
    25%         0.000000
    50%         0.000000
    75%         0.000000
    max      3683.000000
    Name: capital.loss, dtype: float64




```python
df_cleaned[df_cleaned['income'] == 1]['capital.loss'].value_counts().sort_index()
```




    0       6769
    653        2
    1485      28
    1564      24
    1755       2
    1825       3
    1848      50
    1887     155
    1902     181
    1977     162
    2174       6
    2201       1
    2231       3
    2246       6
    2258      13
    2282       1
    2377       8
    2392       8
    2415      45
    2444      12
    2472       1
    2547       4
    2559      12
    2824       8
    3004       1
    3683       1
    Name: capital.loss, dtype: int64




```python
df_cleaned[df_cleaned['income'] == 1]['capital.loss'].value_counts().sort_index()[:10]
```




    0       6769
    653        2
    1485      28
    1564      24
    1755       2
    1825       3
    1848      50
    1887     155
    1902     181
    1977     162
    Name: capital.loss, dtype: int64




```python
double_kde_image('capital.loss')
```


    
![png](output_131_0.png)
    


Перевіряємо гіпотезу про розподіл:


```python
mann_whitneyu('capital.loss')
```

    p-value: 0.18725508559108067.
    

Висновок: розподіл однаковий.

Перевіряємо гіпотезу про середні значення:


```python
ranksums('capital.loss')
```

    p-value: 0.37467793415104444.
    

Висновок: середні значення однакові.

Спробуємо добавити новий показник до нашої таблиці. Це буде різниця між прибутком та витратами.


```python
df_cleaned['rest'] = df_cleaned['capital.gain'] - df_cleaned['capital.loss']
```


```python
hist_image('rest', 15)
```


    
![png](output_140_0.png)
    


"hours.per.week":


```python
hist_image('hours.per.week', 5)
```


    
![png](output_142_0.png)
    



```python
double_kde_image('hours.per.week')
```


    
![png](output_143_0.png)
    


Перевіримо гіпотезу про розподіл:


```python
mann_whitneyu('hours.per.week')
```

    p-value: 0.000279760257625317.
    

Гіпотезу можна відхилити.

Перевіряємо гіпотезу про рівність середніх:


```python
ranksums('hours.per.week')
```

    p-value: 0.0005679373437848791.
    

Висновок: приймаємо альтернативну гіпотезу.

Добавляю ще один показник, який показуватиме відношення оплати житла на місяць відносно всіз витрат.


```python
df_cleaned['hours.per.week'].unique()
```




    array([18, 40, 45, 20, 35, 55, 76, 50, 42, 25, 32, 90, 60, 48, 70, 52, 72,
           39,  6, 65, 80, 67, 99, 30, 75, 12, 26, 10, 84, 38, 62, 44,  8, 28,
           59,  5, 24, 57, 34, 37, 46, 56, 41, 98, 43, 15,  1, 36, 47, 68, 54,
            2, 16,  9,  3,  4, 33, 23, 22, 64, 51, 19, 58, 63, 53, 96, 66, 21,
            7, 13, 27, 14, 77, 31, 78, 11, 49, 17, 85, 87, 88, 73, 89, 97, 94,
           29, 82, 86, 91, 81, 92, 61, 74, 95], dtype=int64)




```python
df_cleaned['capital.loss'].unique()
```




    array([4356, 3900, 3770, 3683, 3004, 2824, 2754, 2603, 2559, 2547, 2472,
           2467, 2457, 2444, 2415, 2392, 2377, 2352, 2339, 2282, 2267, 2258,
           2246, 2238, 2231, 2206, 2205, 2201, 2179, 2174, 2149, 2129, 2080,
           2057, 2051, 2042, 2002, 2001, 1980, 1977, 1974, 1944, 1902, 1887,
           1876, 1848, 1844, 1825, 1816, 1762, 1755, 1741, 1740, 1735, 1726,
           1721, 1719, 1672, 1669, 1668, 1651, 1648, 1628, 1617, 1602, 1594,
           1590, 1579, 1573, 1564, 1539, 1504, 1485, 1411, 1408, 1380, 1340,
           1258, 1138, 1092,  974,  880,  810,  653,  625,  419,  323,  213,
            155,    0], dtype=int64)




```python
def func(x):
    if x == 0:
        return 0
    else:
        return 1 / x

df_cleaned['cap_part_of_house'] = df_cleaned['capital.loss'] / 4 * df_cleaned['hours.per.week']
df_cleaned['cap_part_of_house'] = df_cleaned['cap_part_of_house'].apply(func)
```


```python
hist_image('cap_part_of_house', 5)
```


    
![png](output_154_0.png)
    


Ну і на завершення обробимо місце народження.
Для цього використовую кластерний аналіз.

Дивлюся, які країни є у переліку.


```python
df_cleaned['native.country'].unique()
```




    array(['United-States', 'Mexico', 'Greece', 'Vietnam', 'China', 'Taiwan',
           'India', 'Philippines', 'Trinadad&Tobago', 'Canada', 'South',
           'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran', 'England',
           'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba', 'Ireland',
           'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic', 'Haiti',
           'Hungary', 'Columbia', 'Guatemala', 'El-Salvador', 'Jamaica',
           'Ecuador', 'France', 'Yugoslavia', 'Portugal', 'Laos', 'Thailand',
           'Outlying-US(Guam-USVI-etc)', 'Scotland'], dtype=object)



Створюю окремий датафрейм зі стовпцями income та з індексами назв країн.


```python
lt = np.sort(df_cleaned['native.country'].unique()).tolist()
lt_warn = ['Holand-Netherlands', 'Outlying-US(Guam-USVI-etc)', ]

incomes = df_cleaned.groupby('native.country')['income'].value_counts()

income_count0 = incomes['Cambodia'].values.reshape(1, 2)
df_country = pd.DataFrame(data=income_count0, columns=[0, 1], index=['Cambodia'])

for country in lt[1:]:
    income_count = incomes[country].values
    
    if country in lt_warn:
        income_count = np.concatenate((income_count, np.array(0)), axis=None)

    income_count = income_count.reshape(1, 2)
    income_ser = pd.DataFrame(data=income_count, columns=[0, 1], index=[country])
    df_country = pd.concat([df_country, income_ser], axis=0)
    
df_country
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cambodia</th>
      <td>11</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>71</td>
      <td>36</td>
    </tr>
    <tr>
      <th>China</th>
      <td>48</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Columbia</th>
      <td>54</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Cuba</th>
      <td>67</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Dominican-Republic</th>
      <td>65</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Ecuador</th>
      <td>23</td>
      <td>4</td>
    </tr>
    <tr>
      <th>El-Salvador</th>
      <td>91</td>
      <td>9</td>
    </tr>
    <tr>
      <th>England</th>
      <td>56</td>
      <td>30</td>
    </tr>
    <tr>
      <th>France</th>
      <td>15</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>84</td>
      <td>44</td>
    </tr>
    <tr>
      <th>Greece</th>
      <td>21</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Guatemala</th>
      <td>58</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Haiti</th>
      <td>38</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Holand-Netherlands</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Honduras</th>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Hong</th>
      <td>13</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Hungary</th>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>India</th>
      <td>60</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Iran</th>
      <td>24</td>
      <td>18</td>
    </tr>
    <tr>
      <th>Ireland</th>
      <td>19</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>44</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Jamaica</th>
      <td>70</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>36</td>
      <td>23</td>
    </tr>
    <tr>
      <th>Laos</th>
      <td>15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Mexico</th>
      <td>573</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Nicaragua</th>
      <td>31</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Outlying-US(Guam-USVI-etc)</th>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Peru</th>
      <td>28</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Philippines</th>
      <td>128</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>45</td>
      <td>11</td>
    </tr>
    <tr>
      <th>Portugal</th>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Puerto-Rico</th>
      <td>97</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Scotland</th>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>South</th>
      <td>57</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Taiwan</th>
      <td>23</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Thailand</th>
      <td>14</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Trinadad&amp;Tobago</th>
      <td>16</td>
      <td>2</td>
    </tr>
    <tr>
      <th>United-States</th>
      <td>20478</td>
      <td>6993</td>
    </tr>
    <tr>
      <th>Vietnam</th>
      <td>59</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Yugoslavia</th>
      <td>10</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_country.shape[0]
```




    41



Завантажую перший пакет для обробки


```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

link = linkage(df_country, 'ward', 'euclidean')
```


```python
fig, axes = plt.subplots(1, 1, figsize=(10, 4))

dn = dendrogram(link)

plt.show()
```


    
![png](output_163_0.png)
    


Бачу, що чітко набір може розділитися на два кластери.


```python
df_country['cluster'] = fcluster(link, 10, criterion='distance')
```


```python
df_country.groupby('cluster').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14.750000</td>
      <td>1.750000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.666667</td>
      <td>4.166667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.500000</td>
      <td>18.500000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>21.000000</td>
      <td>5.666667</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>46.000000</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>36.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>29.666667</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>41.500000</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>56.000000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>60.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>71.000000</td>
      <td>36.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>67.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>84.000000</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>94.000000</td>
      <td>10.500000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>67.500000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>57.000000</td>
      <td>3.333333</td>
    </tr>
    <tr>
      <th>19</th>
      <td>57.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>128.000000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>573.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>20478.000000</td>
      <td>6993.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_country.groupby('cluster').size()
```




    cluster
    1     4
    2     6
    3     1
    4     2
    5     3
    6     1
    7     2
    8     1
    9     3
    10    2
    11    1
    12    1
    13    1
    14    1
    15    1
    16    2
    17    2
    18    3
    19    1
    20    1
    21    1
    22    1
    dtype: int64



Для кращого аналізу підключаю метод KMeans і будую графік схилу для підбору оптимального значення кількості кластерів.


```python
from sklearn.cluster import KMeans

Clusters = range(1, 8)

models = [KMeans(n_clusters=n, random_state=1).fit(df_country) for n in Clusters]
dist = [model.inertia_ for model in models]

plt.plot(Clusters, dist, marker='o')
plt.xlabel('Clusters')
plt.ylabel('Sum of distance')
plt.title('The Elbow Method')
plt.show()
```


    
![png](output_169_0.png)
    


Результат графіку підтверджує попередню дендрограму.
Спробуємо навчити метод (для навчання я обрав 5, адже шляхом перевірки з моделлю Логістичної регресії даний показник оптимальний).


```python
model = KMeans(n_clusters=5, random_state=1)
model.fit(df_country)

df_country['cluster'] = model.labels_
df_country.groupby('cluster').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.904762</td>
      <td>5.238095</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20478.000000</td>
      <td>6993.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>573.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.000000</td>
      <td>31.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56.428571</td>
      <td>17.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_country.groupby('cluster').size()
```




    cluster
    0    21
    1     1
    2     1
    3     4
    4    14
    dtype: int64




```python
df_country[df_country['cluster'] == 0].index
```




    Index(['Cambodia', 'Ecuador', 'France', 'Greece', 'Haiti',
           'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'Iran', 'Ireland',
           'Laos', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Portugal',
           'Scotland', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'Yugoslavia'],
          dtype='object')




```python
df_country[df_country['cluster'] == 1].index
```




    Index(['United-States'], dtype='object')




```python
df_country[df_country['cluster'] == 2].index
```




    Index(['Mexico'], dtype='object')




```python
df_country[df_country['cluster'] == 3].index
```




    Index(['El-Salvador', 'Germany', 'Philippines', 'Puerto-Rico'], dtype='object')




```python
df_country[df_country['cluster'] == 4].index
```




    Index(['Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'England',
           'Guatemala', 'India', 'Italy', 'Jamaica', 'Japan', 'Poland', 'South',
           'Vietnam'],
          dtype='object')



Кластеризую данні таблиці.


```python
country = []
for i in range(5):
    country.append(df_country[df_country['cluster'] == i].index.tolist())

def func(x):
    for i, n in enumerate(country):
        if x in n:
            return i
        continue
        
df_cleaned['native.country'] = df_cleaned['native.country'].apply(func)
df_cleaned['native.country'].unique()
```




    array([1, 2, 0, 4, 3], dtype=int64)



# Створення моделей

Перед тим як почати роботу, щодо створення моделей необхідно активувати пакети.


```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
```

Далі треба закінчити обробку даних для прогнозування. Для цього кодую усі номінальні та категоріальні показники.
Для статі використовую бінарний показник.


```python
cls_label = LabelEncoder()

df_cleaned['sex'] = cls_label.fit_transform(df_cleaned['sex'])
```

Категоріальні показники кодую за допомогою векторів та видаляю першого представника для попередження виникнення колінеарності.


```python
model_df = pd.get_dummies(df_cleaned, drop_first=True)
```

Далі розділяю набір даних для моделювання.


```python
X = model_df.drop(columns='income')
y = model_df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
```

Проводжу нормалізацію усіх неперервних ознак (адже всі моделі, окрім лісів, чутливі до масштабу).

Розпочну з навчання моделі Логістичної регресії:


```python
mm = MinMaxScaler() 

X_train_norm = mm.fit_transform(X_train) 
X_test_norm = mm.transform(X_test) 
```


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=300, random_state=1)
lr.fit(X_train_norm, y_train)
```




    LogisticRegression(max_iter=300, random_state=1)



Створюю функцію для перевірки якості моделі:


```python
def score(model_name, model, X_tr, X_t):
    test_score = accuracy_score(y_test, model.predict(X_t)) * 100
    train_score = accuracy_score(y_train, model.predict(X_tr)) * 100

    results_df = pd.DataFrame(data=[[model_name, train_score, test_score]], 
                              columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
    print(results_df)
```


```python
score("Logistic Regression", lr, X_train_norm, X_test_norm)
```

                     Model  Training Accuracy %  Testing Accuracy %
    0  Logistic Regression            84.866736           84.840102
    

Видно, що модель дала 84.87% точності на тренувальному наборі та 84.84% на тестовому.

Спробую покращити показник з допомогою налаштування гіперпараметрів:


```python
from sklearn.model_selection import GridSearchCV 

param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.05, 1, 10, 100, 1000]
param_grid = {'penalty': ['l1','l2'], 'C': param_range}

gs = GridSearchCV(estimator=LogisticRegression(max_iter=500, solver='liblinear', random_state=1),
                 param_grid=param_grid,
                 scoring='accuracy',
                 cv=10,
                 refit=True,
                 n_jobs=-1)

gs.fit(X_train_norm, y_train)

print(f'Найкращий скор: {gs.best_score_}.\n')
print(f'Найкращі параметри моделі: {gs.best_params_}.')
```

    Найкращий скор: 0.849474107084214.
    
    Найкращі параметри моделі: {'C': 10, 'penalty': 'l2'}.
    


```python
new_lr = LogisticRegression(max_iter=500, penalty='l2', solver='liblinear', random_state=1, C=10)
new_lr.fit(X_train_norm, y_train)

score("Logistic Regression", new_lr, X_train_norm, X_test_norm)
```

                     Model  Training Accuracy %  Testing Accuracy %
    0  Logistic Regression            85.056436           85.083545
    

Показники покращилися на обох вибірках.

Далі спробую використати модель методу опорних векторів:


```python
from sklearn.svm import SVC

svc = SVC(kernel="rbf", random_state=1)
svc.fit(X_train_norm, y_train)
```




    SVC(random_state=1)




```python
score("SVC", svc, X_train_norm, X_test_norm)
```

      Model  Training Accuracy %  Testing Accuracy %
    0   SVC            84.487338           84.165099
    

І під кінець, хочеться спробувати модель RandomForestClassifier. 

Но перед використанням знайду оптимальні показники:


```python
from sklearn.ensemble import RandomForestClassifier

n_estimators = np.array([150, 200, 250, 300])
max_features = np.array([3, 4, 5, 6, 7])

param_grid = {'n_estimators': n_estimators, 'max_features': max_features}

grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv=5)
grid.fit(X_train, y_train)

print(grid.best_score_) 
print(grid.best_params_)
```


```python
random_forest = RandomForestClassifier(max_features=5, n_estimators=200, random_state=1)
random_forest.fit(X_train, y_train)

score("RandomForest", random_forest, X_train, X_test)
```

              Model  Training Accuracy %  Testing Accuracy %
    0  RandomForest            97.837428           84.762642
    

Також дана модель дозволяє визначати ступені важливості факторів:


```python
feat_labels = X_train.columns

forest = RandomForestClassifier(max_depth=5, n_estimators=250, random_state=1)
forest.fit(X_train_norm, y_train)
 
importances = forest.feature_importances_ 
indices = np.argsort(importances)[::-1] 
for f in range(X_train.shape[1]): 
    print("%2d %-*s %f" % (f + 1, 30, 
        feat_labels[indices[f]], 
        importances[indices[f]])) 
```

     1 marital.status_Married-civ-spouse 0.208376
     2 rest                           0.162334
     3 capital.gain                   0.129143
     4 education.num                  0.085734
     5 relationship_Relative-child    0.067327
     6 marital.status_Never-married   0.063456
     7 age                            0.055066
     8 relationship_Not-in-family     0.030107
     9 hours.per.week                 0.025373
    10 capital.loss                   0.023189
    11 sex                            0.021535
    12 education_Step                 0.019095
    13 education_Bachelors            0.016638
    14 occupation_Exec-managerial     0.014467
    15 education_School               0.014145
    16 occupation_Prof-specialty      0.013325
    17 lvl_edu                        0.012069
    18 education_Masters              0.009456
    19 relationship_Wife              0.008369
    20 cap_part_of_house              0.004130
    21 education_HS-grad              0.003740
    22 occupation_Other-service       0.003665
    23 workclass_Self-emp-inc         0.002158
    24 education_Assoc-voc            0.001295
    25 occupation_Craft-repair        0.000945
    26 education_Some-college         0.000766
    27 occupation_Farming-fishing     0.000464
    28 workclass_Private              0.000461
    29 marital.status_Widowed         0.000360
    30 race_White                     0.000339
    31 workclass_Self-emp-not-inc     0.000309
    32 occupation_Machine-op-inspct   0.000288
    33 occupation_Handlers-cleaners   0.000281
    34 race_Black                     0.000245
    35 occupation_Sales               0.000208
    36 occupation_Tech-support        0.000185
    37 occupation_Transport-moving    0.000177
    38 native.country                 0.000167
    39 marital.status_Separated       0.000155
    40 workclass_Local-gov            0.000125
    41 marital.status_Married-spouse-absent 0.000085
    42 workclass_State-gov            0.000062
    43 race_Asian-Pac-Islander        0.000059
    44 marital.status_Married-AF-spouse 0.000059
    45 occupation_Protective-serv     0.000038
    46 race_Other                     0.000027
    47 occupation_Priv-house-serv     0.000009
    48 occupation_Armed-Forces        0.000000
    49 workclass_Without-pay          0.000000
    

Можна побачити, які показники найбільше впливають на прогноз класифікаційної моделі.


```python
import session_info

session_info.show()
```




<details>
<summary>Click to view session information</summary>
<pre>
-----
matplotlib          3.3.2
numpy               1.19.2
pandas              1.1.3
scipy               1.5.2
session_info        1.0.0
sklearn             0.23.2
-----
</pre>
<details>
<summary>Click to view modules imported as dependencies</summary>
<pre>
PIL                 8.0.1
backcall            0.2.0
bottleneck          1.3.2
cffi                1.14.3
colorama            0.4.4
cycler              0.10.0
cython_runtime      NA
dateutil            2.8.1
decorator           4.4.2
google              NA
ipykernel           5.3.4
ipython_genutils    0.2.0
jedi                0.17.1
joblib              0.17.0
kiwisolver          1.3.0
mkl                 2.3.0
mpl_toolkits        NA
nt                  NA
ntsecuritycon       NA
numexpr             2.7.1
parso               0.7.0
pickleshare         0.7.5
pkg_resources       NA
prompt_toolkit      3.0.8
psutil              5.7.2
pygments            2.7.2
pyparsing           2.4.7
pythoncom           NA
pytz                2020.1
pywintypes          NA
simplejson          3.17.2
six                 1.15.0
sphinxcontrib       NA
storemagic          NA
threadpoolctl       2.1.0
tornado             6.0.4
traitlets           5.0.5
typing_extensions   NA
wcwidth             0.2.5
win32api            NA
win32com            NA
win32security       NA
zmq                 19.0.2
zope                NA
</pre>
</details> <!-- seems like this ends pre, so might as well be explicit -->
<pre>
-----
IPython             7.19.0
jupyter_client      6.1.7
jupyter_core        4.6.3
jupyterlab          2.2.6
notebook            6.1.4
-----
Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
Windows-10-10.0.19041-SP0
-----
Session information updated at 2021-10-05 21:03
</pre>
</details>


