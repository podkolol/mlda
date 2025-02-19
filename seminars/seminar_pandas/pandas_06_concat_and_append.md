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

# Объединение наборов данных: функция `pd.concat` и метод `pd.DataFrame.append`


Часто при исследовании данных проходится объединять данные из различных источников.
Эти операции могут включать в себя все, что угодно: от очень простой конкатенации двух разных наборов данных до более сложных объединений и слияний в стиле баз данных, которые требуют корректной обработки любых перекрытий между наборами данных.
`Series` и `DataFrame` поддерживают такие операции, включая функции и методы, которые делают такую обработку данных быстрой и простой.

Рассмотрим простую конкатенацию `Series` и `DataFrame` с помощью функции `pd.concat`.

Начнем со стандартного импорта:

```python
import pandas as pd
import numpy as np
```

Для удобства дальнейшей работы определим функцию, которая создает `DataFrame` необходимой формы:

```python jupyter={"outputs_hidden": false}
def make_df(cols, ind):
    '''Создание DataFrame заданной формы'''
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

make_df('ABC', range(3))
```

Кроме этого, создадим класс, который позволит отображать несколько `DataFrame` бок о бок. 
Код использует специальный метод `_repr_html_`, который IPython использует для реализации отображения объектов:

```python
class Display:
    '''Создает HTML-представление нескольких объектов'''
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_()) for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a)) for a in self.args)


df1 = make_df('ABC', range(3))
df2 = make_df('CDE', range(3))

Display('df1', 'df2')
```

## Конкатенация объектов `Series` и `DataFrame` 

Конкатенация объектов `Series` и `DataFrame` очень похожа на конкатенацию массивов Numpy, которую можно выполнить с помощью функции `np.concatenate`, как обсуждалось в [Основы работы с массивами NumPy](numpy_02_the_basics_of_numpy_arrays.md).
Напомним, что с его помощью можно объединить содержимое двух или более массивов в один массив:

```python jupyter={"outputs_hidden": false}
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])
```

Первый аргумент &mdash; список или кортеж массивов для конкатенации.
Кроме того, он принимает ключевое слово `axis`, которое позволяет указать ось, вдоль которой будет объединен результат:

```python jupyter={"outputs_hidden": false}
x = [[1, 2],
     [3, 4]]
np.concatenate([x, x], axis=1)
```

## Простая конкатенация с `pd.concat`

<!-- #region -->
В Pandas используется функция `pd.concat()`, которая имеет синтаксис, похожий на `np.concatenate`, но содержит ряд дополнительных опций, которые обсудим чуть позже:

```python
# Signature in Pandas v2.2
pd.concat(
    objs: 'Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame]',
    *,
    axis: 'Axis' = 0,
    join: 'str' = 'outer',
    ignore_index: 'bool' = False,
    keys: 'Iterable[Hashable] | None' = None,
    levels=None,
    names: 'list[HashableT] | None' = None,
    verify_integrity: 'bool' = False,
    sort: 'bool' = False,
    copy: 'bool | None' = None,
) -> 'DataFrame | Series'
```

Метод `pd.concat()` можно использовать для простой конкатенации объектов `Series` или `DataFrame`, так же как `np.concatenate()` можно использовать для простой конкатенации массивов:
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])
```

Он также работает для объединения многомерных объектов, таких как `DataFrame`:

```python jupyter={"outputs_hidden": false}
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([df1, df2])')
```

По умолчанию конкатенация проводится построчно внутри `DataFrame` (т.е. `axis=0`).
Как и `np.concatenate`, `pd.concat` позволяет указать ось, вдоль которой будет выполняться конкатенация.
Рассмотрим следующий пример:

```python jupyter={"outputs_hidden": false}
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
display('df3', 'df4', "pd.concat([df3, df4], axis='columns')")
```

Запись `axis='columns` является интуитивно понятным эквивалентом записи `axis=1`.


### Дублирующиеся индексы

Одним из важных различий между `np.concatenate` и `pd.concat` является то, что конкатенация Pandas *сохраняет индексы*, даже если в результате некоторые индексы будут дублироваться!
Рассмотрим простой пример:

```python jupyter={"outputs_hidden": false}
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index  # сделайте дубликаты индексов!
display('x', 'y', 'pd.concat([x, y])')
```

Обратите внимание на повторяющиеся индексы в результате.
Хотя в объектах `DataFrame` это и допустимо, но подобный результат часто может быть нежелателен. 
`pd.concat()` предлагает несколько способов справиться с этой проблемой.


#### Отслеживание повторов как ошибок

Если необходимо просто проверить, что индексы в результате `pd.concat()` не перекрываются, можно указать флаг `verify_integrity`.
Если задано значение True, конкатенация вызовет исключение, если имеются дублирующиеся индексы.
Вот пример, где для ясности перехватим и выведем сообщение об ошибке:

```python jupyter={"outputs_hidden": false}
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)
```

#### Игнорирование индекса

Иногда сам индекс не имеет значения, можно его просто игнорировать.
Эту опцию можно указать с помощью флага `ignore_index`.
Если для этого параметра установлено значение True, конкатенация создаст новый целочисленный индекс для результирующего объекта:

```python jupyter={"outputs_hidden": false}
display('x', 'y', 'pd.concat([x, y], ignore_index=True)')
```

#### Добавление ключей MultiIndex

Другой вариант &mdash; использовать опцию `keys` для указания меток для источников данных. 
Результатом будет иерархически индексированный `DataFrame`:

```python jupyter={"outputs_hidden": false}
display('x', 'y', 'pd.concat([x, y], keys=["x", "y"])')
```

Результатом является мультииндексированный объект `DataFrame`, для работы с которым можно использовать инструменты, обсуждаемые в [Иерархическая индексация](pandas_05_hierarchical_indexing.md), для преобразования этих данных в нужное представление.


### Конкатенация с использованием соединений

В рассмотренных примерах, в основном объединялись `DataFrame` с общими названиями столбцов.
На практике данные из разных источников могут иметь разные наборы имен столбцов, и `pd.concat` предлагает в этом случае несколько вариантов.
Рассмотрим объединение следующих двух `DataFrame`, которые имеют некоторые (но не все!) cтолбцы с одинаковыми названиями:

```python jupyter={"outputs_hidden": false}
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
display('df5', 'df6', 'pd.concat([df5, df6])')
```

По умолчанию записи, для которых отсутствуют данные, заполняются NA-значениями.
Чтобы изменить это, можно указать один из нескольких вариантов для параметров `join` и `join_axes` функции конкатенации.
По умолчанию соединение представляет собой объединение входных столбцов (`join='outer'`), но можно изменить это поведение на пересечение столбцов, используя `join='inner'`:

```python jupyter={"outputs_hidden": false}
display('df5', 'df6', 'pd.concat([df5, df6], join="inner")')
```

Другой вариант &mdash; переиндексировать столбцы для достижения необходимого результата.
Здесь мы укажем, что возвращаемые столбцы должны быть такими же, как и в первом вводе:

```python jupyter={"outputs_hidden": false}
display('df5', 'df6', 'pd.concat([df5, df6.reindex(columns = df5.columns)])')
```

Сочетание опций функции `pd.concat` допускает широкий спектр возможных вариантов поведения при объединении двух наборов данных.
Помните об этом, когда будете использовать эти инструменты для своих собственных данных.


В следующем разделе рассмотрим еще один более мощный подход к объединению данных из нескольких источников &mdash; слияния/объединения в стиле баз данных, реализованные в `pd.merge`.
Дополнительную информацию о `concat()` и связанных с ним функциях смотрите в разделе ["Merge, join, concatenate and compare" section](http://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) документации Pandas.
