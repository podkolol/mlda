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

# Гистограммы, разбиения по интервалам (биннинги) и плотность


Простая гистограмма может стать отличным первым шагом в понимании набора данных.


```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
plt.style.use('default')
```

```python
data = np.random.randn(1000)
```

```python jupyter={"outputs_hidden": false}
plt.hist(data);
```

Функция `hist()` имеет множество опций для настройки как расчета, так и отображения:

```python jupyter={"outputs_hidden": false}
plt.hist(data, bins=30, density=True, alpha=0.5,
         histtype='stepfilled', color='steelblue', 
         edgecolor='none');
```

В документации `plt.hist` содержится дополнительная информация о других доступных параметрах настройки.


При необходимости сравнения гистограмм нескольких распределений очень полезным может оказаться параметр `histtype='stepfilled'` с заданной прозрачностью `alpha`:

```python jupyter={"outputs_hidden": false}
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);
```

<!-- #raw -->
Если нужно просто вычислить гистограмму (то есть подсчитать количество точек в заданном интервале), а не отображать ее, то лучше воспользоваться функцией `np.histogram()`:
<!-- #endraw -->

```python jupyter={"outputs_hidden": false}
counts, bin_edges = np.histogram(data, bins=5)
print(counts)
```

## Двумерные гистограммы и разбиения по интервалам

Аналогично созданию одномерных гистограмм, можно создавать гистограммы в двух измерениях, распределяя точки по двумерным интервалам. 
Кратко рассмотрим несколько способов сделать это.
Начнем с определения некоторых данных &mdash; массивов `x` и `y`, взятых из многомерного гауссовского распределения:

```python jupyter={"outputs_hidden": false}
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
```

### Двумерная гистограмма с помощью `plt.hist2d`:

Одним из простых способов построения двумерной гистограммы является использование функции `plt.hist2d`:

```python jupyter={"outputs_hidden": false}
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('Количество точек в интервале')
```

Как и `plt.hist`, `plt.hist2d` имеет ряд дополнительных опций для точной настройки графика и группировки точек по интервалам, которые подробно описаны в документации.
`plt.hist2d` также имеет аналог `np.histogram2d`, который можно использовать для расчета гистограммы без ее отображения:

```python jupyter={"outputs_hidden": false}
counts, xedges, yedges = np.histogram2d(x, y, bins=30)
```

Для расчета и отображения гистограмм в более чем двух измерениях используется функция `np.histogramdd`.


### Гексагональное разбиение по интервалам с помощью функции `plt.hexbin`

Двумерная гистограмма создает мозаику квадратами вдоль осей координат.
Другой естественной формой для такого мозаичного представления является правильный шестиугольник.
Для этой цели Matplotlib предоставляет процедуру `plt.hexbin`, которая визуализирует двумерный набор данных, размещенный в сетке шестиугольников:

```python jupyter={"outputs_hidden": false}
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='Количество точек в интервале')
```

У функции `plt.hexbin` имеется множество интересных параметров, включая возможность задавать вес для каждой точки и менять выводимое значение для каждого интервала на любой сводный показатель библиотеки NumPy (среднее значение весов, стандартное отклонение весов и т. д.).


### Ядерная оценка плотности распределения

Другим распространенным методом оценки плотности в нескольких измерениях является *ядерная оценка плотности распределения* (*Kernel Density Estimation*, *KDE*).
KDE можно рассматривать как способ &laquo;размазать&raquo; точки в пространстве и сложить результаты для получения гладкой функции.
В пакете `scipy.stats`реализована одна из самых быстрых и простых реализаций KDE.
Вот краткий пример использования KDE для тех же данных, чтоо и в предудущих примерах:

```python jupyter={"outputs_hidden": false}
from scipy.stats import gaussian_kde

# Подбор на массиве размера [Ndim, Nsamples]
data = np.vstack([x, y])
kde = gaussian_kde(data)

# Вычисление на регулярной координатной сетке
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# Вывод графика результата в виде изображения
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("Плотность")
```

В KDE имеется параметр *длина сглаживания*, который позволяет эффективно выбирать компромисс между гладкостью и детализацией(один из примеров повсеместного компромисса между смещением и дисперсией).
Литература по выбору подходящей длины сглаживания обширна:  в функции `gaussian_kde` используется эмпирическое правило для поиска квазиоптимальной длины сглаживания для входных данных.
