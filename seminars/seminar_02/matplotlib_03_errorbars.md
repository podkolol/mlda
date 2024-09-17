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

# Визуализация погрешностей


Для любого научного измерения точный учет ошибок почти так же важен, если не важнее, чем точное сообщение самого числа.
При визуализации данных и результатов эффективное отображение этих ошибок позволит передать с помощью графика намного более полную информацию.


## Простые планки погрешностей


Начнем с импорта необходимых пакетов и настройки блокнота для построения графиков:

```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
plt.style.use('default')
```

Простые планки погрешностей можно создать с помощью вызова всего одной функции Matplotlib:

```python jupyter={"outputs_hidden": false}
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='ok');
```

Здесь `fmt` &mdash; это код формата, управляющий внешним видом линий и точек.
Его синтаксис совпадает с синтаксисом, используемым в `plt.plot`, описанное в [Простые линейные графики](Простые линейные графики).

Допонительно можно создавать горизонтальные планки погрешностей (`xerr`), односторонние планки погрешностей и много других вариантов.
Подробнее узнать об имеющихся опциях, можно в документации по функции `plt.errorbar`.

```python jupyter={"outputs_hidden": false}
plt.errorbar(x, y, yerr=dy, 
             fmt='o', 
             color='black',
             ecolor='lightgray', 
             elinewidth=3, 
             capsize=0);
```

В дополнение к этим параметрам также можно указать горизонтальные полосы погрешностей (`xerr`), односторонние полосы погрешностей и многие другие варианты.
Более подробную информацию о доступных параметрах см. в строке документации ``plt.errorbar``.


## Непрерывные погрешности

В некоторых ситуациях желательно отображать планки погрешностей для непрерывных величин.
Хотя Matplotlib не имеет встроенной удобной процедуры для такого типа приложений, сравнительно легко объединить примитивы, такие как `plt.plot` и `plt.fill_between`, для получения нужного результата.

Выполним простую *регрессию на основе Гауссова процесса* (*Gaussian process regression*, *GPR*), используя API пакета Scikit-Learn.
Данный метод представляет собой метод подбора по имеющимся данным очень гибкой непараметрической функции с непрерывной мерой неопределенности измерения.
Не будем пока углубляться в детали метода, а просто сосредоточимся на том, как можно визуализировать такие непрерывные погрешности:

```python
# "Получаем" истинные данные и отрисовываем их
X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Истинный процесс");
```

```python
# Для обучения регрессии на основе Гауссовского процесса выберем только несколько точек
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]
X_train
```

Теперь применим гауссовский процесс к этим нескольким точкам обучающей выборки.
Будем использовать ядро радиальной базисной функции (RBF) и постоянный параметр для подгонки амплитуды.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_
```

Теперь используем ядро для вычисления среднего прогноза для всего набора данных и построения 95% доверительного интервала.

```python
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$ (Истинные данные)", linestyle="dotted")
plt.scatter(X_train, y_train, label="Точки обучающей выборки")
plt.plot(X, mean_prediction, label="Предсказанные данные")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% доверительный интервал",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("регрессию на основе Гауссова процесса");
```

В областях возле измеренной точки данных модель жестко ограничена, что и отражается в малых ошибках модели. В удаленных же от измеренной
точки данных областях модель жестко не ограничивается и ошибки модели растут.

Дополнительную информацию о доступных параметрах `plt.fill_between()` (и тесно связанной с ней функции `plt.fill()`) смотрите в документации.

Еще больше возможностей по визуализации погрешностей предоставляет пакет Seaborn.
