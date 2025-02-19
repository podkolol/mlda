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

# Настройка Matplotlib: конфигурации и таблицы стилей

```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
plt.style.use('default')
```

## Настройка сюжета вручную

Пусть, например, имеется вот такая довольно унылая гистограмма по умолчанию:

```python jupyter={"outputs_hidden": false}
x = np.random.randn(1000)
plt.hist(x);
```

Настроим ее вид вручную, превратив эту гистограмму в намного более приятный глазу график:

```python jupyter={"outputs_hidden": false}
# используем серый фон
ax = plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)

# рисуем сплошные белые линии сетки
plt.grid(color='w', linestyle='solid')

# скрываем основные линии осей координат
for spine in ax.spines.values():
    spine.set_visible(False)
    
# скрываем деления сверху и справа
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# осветляем цвет делений и меток
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
# задаем цвет заливки и границ гистограммы
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');
```

Выглядит намного лучше, но, для таких настроек потребовалось немало труда и не хотелось бы снова проделывать их все при каждом создании графика. 
К счастью, существует способ задать эти настройки один раз для всех графиков.


## Изменение значений по умолчанию: ``rcParams``

Каждый раз при загрузке Matplotlib определяется конфигурация времени выполнения (runtime config, rc), содержащая стили по умолчанию для каждого создаваемого элемента графика.
Эту конфигурацию можно настроить с помощью удобной утилиты `plt.rc`.
Давайте посмотрим, как будет выглядеть изменение параметров rc, чтобы график по умолчанию выглядел так же, как настроенный график выше.

Начнем с сохранения копии текущего словаря `rcParams`, чтобы можно было легко сбросить эти изменения в текущем сеансе:

```python
IPython_default = plt.rcParams.copy()
```

Теперь можно использовать функцию `plt.rc` для изменения некоторых из этих настроек:

```python jupyter={"outputs_hidden": false}
from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
```

Переопределив необходимые настройки, можем создать график и увидеть их в действии:

```python jupyter={"outputs_hidden": false}
plt.hist(x);
```

Давайте посмотрим, как выглядят простые линейные графики с этими параметрами rc:

```python jupyter={"outputs_hidden": false}
for i in range(4):
    plt.plot(np.random.rand(10))
```

Эти настройки можно сохранить в файле *.matplotlibrc*, о котором вы можете прочитать в [документации Matplotlib](http://Matplotlib.org/users/customizing.html).


## Таблицы стилей

Еще один способ настраивать Matplotlib &mdash; использовать его таблицы стилей.

В релизе Matplotlib версии 1.4 в августе 2014 года был добавлен очень удобный модуль `style`, который включает ряд новых таблиц стилей по умолчанию, а также возможность создавать и упаковывать собственные стили. 
Эти таблицы стилей форматируются аналогично файлам *.matplotlibrc*, упомянутым ранее, но должны иметь расширение *.mplstyle*.

Таблицы стилей, включенные по умолчанию, чрезвычайно полезны, даже если нет необходимости создавать собственный стиль.

Доступные стили перечислены в `plt.style.available` &mdash; приведем для для краткости только первые пять:

```python jupyter={"outputs_hidden": false}
plt.style.available[:5]
```

<!-- #region -->
Основной способ переключиться на таблицу стилей &mdash; вызвать

``` python
plt.style.use('stylename')
```

Но следует иметь в виду, что это изменит стиль на весь остаток сеанса работы!
В качестве альтернативы можно использовать менеджер контекста стилей, который временно устанавливает стиль:

``` python
with plt.style.context('stylename'):
    make_a_plot()
```
<!-- #endregion -->

Создадим функцию, которая будет строить два простейших вида графиков:

```python
def hist_and_lines():
    np.random.seed(42)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')
```

Воспользуемся ею, чтобы изучить, как выглядят эти графики с использованием различных встроенных стилей.


### Стиль по умолчанию

Начнем со стиля по умолчанию.
Но сначала восстановим конфигурацию среды выполнения в блокноте до значений по умолчанию:

```python jupyter={"outputs_hidden": false}
# Сброс rcParams
plt.rcParams.update(IPython_default);
```

Теперь посмотрим, как будут выглядеть графики:

```python jupyter={"outputs_hidden": false}
hist_and_lines()
```

### Стиль FiveThiryEight

Стиль `fivethirtyeight` имитирует графику популярного сайта [FiveThirtyEight website](https://fivethirtyeight.com).
 Как можно видеть, он включает жирные
шрифты, толстые линии и прозрачные оси координа:

```python jupyter={"outputs_hidden": false}
with plt.style.context('fivethirtyeight'):
    hist_and_lines()
```

### ggplot

Пакет `ggplot` в языке R &mdash; очень популярный инструмент визуализации.
Стиль `ggplot` пакета Matplotlib имитирует стили по умолчанию из этого пакета:

```python jupyter={"outputs_hidden": false}
with plt.style.context('ggplot'):
    hist_and_lines()
```

### Стиль &laquo;Байесовские методы для хакеров&raquo;

Существует замечательная онлайн-книга &laquo;Вероятностное программирование и байесовские методы для хакеров&raquo;» [Probabilistic Programming and Bayesian Methods for
Hackers](https://dataorigami.net/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/). Она содержит рисунки, созданные с помощью библиотеки Matplotlib, и использует в книге для создания единообразного и приятного внешне стиля набор параметров rc. Этот стиль воспроизведен в таблице стилей `bmh`:

```python jupyter={"outputs_hidden": false}
with plt.style.context('bmh'):
    hist_and_lines()
```

### Темный фон

Для рисунков, используемых в презентациях, часто бывает полезно использовать темный, а не светлый фон.
Стиль `dark_background` предназначен как раз для этого:

```python jupyter={"outputs_hidden": false}
with plt.style.context('dark_background'):
    hist_and_lines()
```

### Оттенки серого

Если нужно подготовить иллюстрации для печатного издания черно-белые иллюстрации, очень полезным будет `grayscale`:

```python jupyter={"outputs_hidden": false}
with plt.style.context('grayscale'):
    hist_and_lines()
```

### Стиль Seaborn

В Matplotlib есть даже таблицы стилей, вдохновленные библиотекой Seaborn.
Эти стили загружаются автоматически при импорте Seaborn в блокнот.

```python jupyter={"outputs_hidden": false}
import seaborn
hist_and_lines()
```

Благодаря всем этим встроенным вариантам построения графиков Matplotlib становится гораздо более полезным как для интерактивной визуализации, так и для создания рисунков для публикации.
