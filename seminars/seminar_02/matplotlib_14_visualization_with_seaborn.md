---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Визуализация с Seaborn


Matplotlib зарекомендовал себя как невероятно полезный и популярный инструмент визуализации, но даже опытные пользователи признают, что он часто оставляет желать лучшего.
Существует несколько обоснованных жалоб на Matplotlib, которые часто возникают:

- API Matplotlib относительно низкоуровневый. Сложная статистическая визуализация возможна, но часто требует *много* шаблонного кода.
- Matplotlib появился более чем на десятилетие раньше Pandas, и поэтому не предназначен для использования с `DataFrame` Pandas. Чтобы визуализировать данные из `DataFrame` Pandas, необходимо извлечь каждую `Series` и часто конвертировать их в нужный формат. Было бы удобнее иметь библиотеку построения графиков, которая может использовать данные из `DataFrame` для построения графиков напрямую.

Решением этой проблемы является [Seaborn](http://seaborn.pydata.org/).
Seaborn предоставляет API поверх Matplotlib, который предлагает разумный выбор для стиля графика и цветов по умолчанию, определяет простые высокоуровневые функции для распространенных типов статистических графиков и интегрируется с функциональностью, предоставляемой Pandas `DataFrame`.

Справедливости ради следует отметить, что команда Matplotlib решает эту проблему.
Но по всем вышеперечисленным причинам Seaborn остается чрезвычайно полезным дополнением.


## Seaborn против Matplotlib

Ниже приведен пример простого графика случайных процессов в Matplotlib с использованием форматирования и цветов графика по умолчанию.
Начнем с типичного импорта:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline
plt.style.use('default')

```

Теперь создадим некоторые случайные данные:

```python
# Создайте некоторые данные
rng = np.random.RandomState(42)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)
```

И сделаем простой график:

```python jupyter={"outputs_hidden": false}
# Постройте график данных с использованием настроек Matplotlib по умолчанию
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
```

Хотя результат содержит всю информацию, которую необходимо было передать, он сделан не слишком эстетично и даже выглядит немного старомодно в контексте визуализации данных XXI века.

Теперь давайте посмотрим, как это работает с Seaborn.
Seaborn имеет множество собственных высокоуровневых процедур построения графиков, но он также может перезаписывать параметры Matplotlib по умолчанию и, в свою очередь, заставлять даже простые скрипты Matplotlib выдавать гораздо более качественные выходные данные.
Задать стиль можно с помощью метода `set()` библиотеки
Seaborn. 
По принятым соглашениям Seaborn импортируется под псевдонимом `sns`:

```python jupyter={"outputs_hidden": false}
import seaborn as sns
sns.set()
```

Теперь давайте повторим те же две строки, что и раньше:

```python jupyter={"outputs_hidden": false}
# тот же код построения графика, что и выше!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
```

Немного лучше.


## Типы графиков библиотеки Seaborn

Основная идея Seaborn заключается в том, что он предоставляет высокоуровневые команды для создания различных типов графиков, полезных для статистического исследования данных, и даже, для некоторой подгонки статистических моделей.

Давайте рассмотрим несколько наборов данных и типов графиков, доступных в Seaborn.
Обратите внимание, что все нижеследующее *можно* сделать с помощью чистого Matplotlib (это, по сути, то, что Seaborn делает под капотом), но API Seaborn гораздо удобнее.


### Гистограммы, KDE и плотности

Часто при визуализации статистических данных все, что вам нужно, &mdash; это построить гистограммы и совместные распределения переменных.
В Matplotlib это относительно просто:

```python jupyter={"outputs_hidden": false}
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], density=True, alpha=0.5)
```

Вместо гистограммы можно получить сглаженную оценку распределения, используя оценку плотности ядра, которую Seaborn делает с помощью `sns.kdeplot`:

```python jupyter={"outputs_hidden": false}
for col in 'xy':
    sns.kdeplot(data[col], fill=True)
```

Гистограммы и KDE можно объединить с помощью `histplot`:

```python jupyter={"outputs_hidden": false}
sns.histplot(data['x'], kde=True)
sns.histplot(data['y'], kde=True);

```

Если передать функции `kdeplot` весь двумерный набор данных, можно получить двумерную визуализацию данных:

```python jupyter={"outputs_hidden": false}
sns.kdeplot(data);
```

Посмотреть на совместное распределение и частные распределения можно, воспользовавшись функцией `sns.jointplot`. 
Для этого графика зададим стиль с белым фоном:

```python jupyter={"outputs_hidden": false}
with sns.axes_style('white'):
    sns.jointplot(data, x='x', y="y",  kind='kde');
```

Существуют и другие параметры, которые можно передать в `jointplot` &mdash; например, можно использовать гексагональную гистограмму:

```python jupyter={"outputs_hidden": false}
with sns.axes_style('white'):
    sns.jointplot(data, x="x", y="y", kind='hex')
```

### Парные графики

При обобщаении графиков совместных распрелений на наборы данных более высоких размерностей, становится очевидным полезность парных графиков (pair plots). 
Они очень удобны для изучения зависимостей между многомерными данными, когда необходимо построить график всех пар значений.

Продемонстрируем это на примере известного набора данных Iris, в котором приведены размеры лепестков и чашелистиков трех видов ирисов:

```python jupyter={"outputs_hidden": false}
iris = sns.load_dataset("iris")
iris.head()
```

Визуализировать многомерные отношения между образцами очень просто:

```python jupyter={"outputs_hidden": false}
sns.pairplot(iris, hue='species', height=2.5);
```

### Фасетные гистограммы

Иногда лучший способ просмотра данных &mdash; гистограммы подмножеств. 
`FacetGrid` от Seaborn делает построение подобной диаграммы чрезвычайно простым делом.
Давайте рассмотрим данные, показывающие размер чаевых, которые получает персонал ресторана, на основе различных показателей:

```python jupyter={"outputs_hidden": false}
tips = sns.load_dataset('tips')
tips.head()
```

```python jupyter={"outputs_hidden": false}
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));
```

```python jupyter={"outputs_hidden": false}
with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("День", "Total Bill");
```

### Совместные распределения

Подобно парному графику можно использовать `sns.jointplot`, чтобы показать совместное распределение между различными наборами данных,  а также соответствующих частных распределений:

```python jupyter={"outputs_hidden": false}
with sns.axes_style('white'):
    sns.jointplot(data=tips, x='total_bill', y='tip',  kind='hex')
```

Совместный график может даже выполнять некоторую автоматическую оценку плотности ядра и регрессию:

```python jupyter={"outputs_hidden": false}
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg');
```

### Столбчатые диаграммы

Временные ряды можно построить с помощью `sns.factorplot`. 
В следующем примере воспользуемся набором данных Planets, доступном в Seaborn:

```python jupyter={"outputs_hidden": false}
planets = sns.load_dataset('planets')
planets.head()
```

```python jupyter={"outputs_hidden": false}
with sns.axes_style('white'):
    g = sns.catplot(data=planets, x='year', aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)
```

Получить еще больше информации можно, если посмотреть на метод, с помощью которого была
открыта каждая из этих планет:

```python jupyter={"outputs_hidden": false}
with sns.axes_style('white'):
    g = sns.catplot(data=planets, x="year", aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Количество открытых планет')
```

Для получения дополнительной информации о построении графиков с помощью Seaborn обратитесь к  [документации Seaborn](http://seaborn.pydata.org/), [руководству по Seaborn](http://seaborn.pydata.org/tutorial.htm) и [галерее Seaborn](http://seaborn.pydata.org/examples/index.html).
