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

# Иерархическая индексация

<!-- #region deletable=true editable=true -->
До этого момента рассматривались в основном на одномерных и двумерных данных, хранящихся в объектах Pandas `Series` и `DataFrame` соответственно.
Часто бывает полезно выйти за пределы двух измерений и хранить данные более высокой размерности, то есть данные, индексированные более чем по двум ключам.
Хотя Pandas предоставляет объекты `Panel` и `Panel4D`, которые изначально обрабатывают трехмерные и четырехмерные данные, на практике гораздо более распространенной схемой является использование *иерархического индексирования* (*hierarchical indexing*),(также известного как *мультииндексирование*  (*multi-indexing*),) для включения в один уровень нескольких *уровней*.
Таким образом, многомерные данные могут быть компактно представлены в одномерных объектах `Series` и двумерных объектах `DataFrame`.

В этом разделе рассмотрим создание объектов `MultiIndex`, соображения по индексированию, срезу и вычислению статистики по множественно индексированным данным, а также полезные процедуры для преобразования между простыми и иерархически индексированными представлениями данных.

Начнем со стандартного импорта:
<!-- #endregion -->

```python deletable=true editable=true
import pandas as pd
import numpy as np
```

<!-- #region deletable=true editable=true -->
## Мультииндексированный объект Series

Давайте начнем с рассмотрения того, как можно представить двумерные данные в одномерном объекте `Series`.
Для конкретности рассмотрим ряд данных, где каждая точка имеет символьный и числовой ключ.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### Плохой способ: ручная фильтрация

Предположим, необходимо отслеживать данные о городах за два разных года.
Используя инструменты Pandas, которые мы уже рассмотрели, у вас может возникнуть соблазн просто использовать кортежи Python в качестве ключей:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
index = [('Москва', 2024), ('Москва', 2022),
         ('Санкт-Петербург', 2024), ('Санкт-Петербург', 2022),
         ('Новосибирск', 2024), ('Новосибирск', 2022), 
         ('Екатеринбург', 2024), ('Екатеринбург', 2022),
         ('Казань', 2024), ('Казань', 2022),
         ('Красноярск', 2024), ('Красноярск', 2022)]

populations = [13149803, 13015126,
               5597763, 5607916,
               1633851, 1636131,
               1536183, 1547044,
               1318604, 1309324	,
               1205473, 1223488]

pop = pd.Series(populations, index=index)
pop
```

<!-- #region deletable=true editable=true -->
Используя эту схему индексации, можно  непосредственно индексировать или выполнять срез ряда данных на основе такого мультииндекса::
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop[('Новосибирск', 2024):('Казань', 2022)]
```

<!-- #region deletable=true editable=true -->
Но на этом удобство заканчивается. 
Например, если нужно выбрать все значения за 2022 год, то придется проделать некоторую громоздкую (и потенциально медленную) работу по фильтрации данных:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop[[i for i in pop.index if i[1] == 2022]]
```

<!-- #region deletable=true editable=true -->
Это конечно приводит к желаемому результату, но не так просто (и гораздо менее эффективно для больших наборов данных), как синтаксис срезов Pandas.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### Лучший способ: Pandas MultiIndex
К счастью, Pandas предлагает лучший способ.
Используемая индексация на основе кортежей по сути является элементарным мультииндексом, а тип `MultiIndex` библиотеки Pandas предоставляет тип операций, который необходим в этой ситуации.
Создать мультииндекс из кортежей можно следующим образом:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
index = pd.MultiIndex.from_tuples(index)
index
```

<!-- #region deletable=true editable=true -->
Проиндексировав заново данные по городам с помощью `MultiIndex`, получим иерархическое представление данных:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop = pop.reindex(index)
pop
```

<!-- #region deletable=true editable=true -->
Здесь первые два столбца представления `Series` содержат множественные значения индекса, а третий сожержит данные.
Обратите внимание, что в первом столбце отсутствуют некоторые записи: в таком многоиндексном представлении любая пустая запись указывает на то же значение, что и строка над ней.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
Теперь, для того чтобы получить доступ ко всем данным, для которых второй индекс равен 2022 году, можно просто использовать нотацию среза Pandas:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop[:, 2022]
```

<!-- #region deletable=true editable=true -->
Результатом является одноиндексный массив, содержащий только те ключи, которые необходимы.
Этот синтаксис гораздо удобнее (а операция гораздо эффективнее!), чем самодельное решение с множественной индексацией на основе кортежей, рассмотренное выше.

Обсудим подробнее подобные операции индексации над иерархически индексированными данными
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### MultiIndex как дополнительное измерение

Можно было бы легко сохранить те же данные, используя простой `DataFrame` с индексом и метками столбцов.
На самом деле, Pandas создан с учетом этой эквивалентности. 
Метод `unstack()` быстро преобразует мультииндексный объект `Series` в индексированный обычным способом `DataFrame`:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop_df = pop.unstack()
pop_df
```

<!-- #region deletable=true editable=true -->
Метод `stack()` обеспечивает противоположную операцию:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop_df.stack()
```

<!-- #region deletable=true editable=true -->
Тогда зачем вообще беспокоиться об иерархической индексации.
Причина проста: мультииндексное представление используется не только для представления двумерных данных в одномерном объекте `Series`, но и для представления данных из трех и более измерений в `Series` или `DataFrame`.
Каждый дополнительный уровень в мультииндексе представляет дополнительное измерение данных.
Использование этого свойства дает гораздо больше гибкости в представлении типов данных.
Например, можно добавить еще один столбец демографических данных для каждого города в каждый год (например, население моложе 18 лет).
С `MultiIndex` это так же просто, как добавить еще один столбец в `DataFrame`:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop_df = pd.DataFrame({'total': pop,
                       'under18': [3297450, 3253781,
                                   1398440, 1402979,
                                   418462,  421032, 
                                   379045,  375761,
                                   327651,  328331,
                                   305368,  307872]})
pop_df
```

<!-- #region deletable=true editable=true -->
Кроме того, все универсальные и другие функции<!--,  обсуждаемые в [Вычисления на массивах NumPy: универсальные функции](numpy_03_computation_on_arrays_ufuncs.md), --> также  отлично работают с иерархическими индексами.
Вычислим долю людей моложе 18 лет по годам, учитывая приведенные выше данные:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
f_u18 = pop_df['under18'] / pop_df['total']
f_u18.unstack()
```

<!-- #region deletable=true editable=true -->
Это дает нам возможность легко и быстро манипулировать даже многомерными данными и исследовать их.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
## Методы создания мультииндекса

Самый простой способ построить мультииндексированных объектов `Series` или `DataFrame` &mdash; это просто передать конструктору список из двух или более индексных массивов. Например:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
df
```

<!-- #region deletable=true editable=true -->
Работа по созданию `MultiIndex` выполняется в фоновом режиме.

Аналогично, если передать словарь с соответствующими кортежами в качестве ключей, Pandas автоматически распознает это и будет использовать `MultiIndex` по умолчанию:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
data = {('Москва', 2024): 13149803, 
         ('Москва', 2022): 13015126,
         ('Санкт-Петербург', 2024): 5597763, 
         ('Санкт-Петербург', 2022): 5607916,
         ('Новосибирск', 2024): 1633851, 
         ('Новосибирск', 2022): 1636131,
         ('Екатеринбург', 2024): 1536183, 
         ('Екатеринбург', 2022): 1547044,
         ('Казань', 2024): 1318604, 
         ('Казань', 2022): 1309324,
         ('Красноярск', 2024): 1205473, 
         ('Красноярск', 2022): 1223488}

pd.Series(data)
```

<!-- #region deletable=true editable=true -->
Тем не менее, иногда бывает полезно явно создать `MultiIndex`
Рассмотрим несколько предназначенных для этого методов.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### Явные конструкторы MultiIndex

Для большей гибкости в построении индекса можно использовать конструкторы методов класса, доступные в `pd.MultiIndex`.
Например, можно создать `MultiIndex` из простого списка массивов, содержащих значения индекса на каждом уровне:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
```

<!-- #region deletable=true editable=true -->
Его можно построить из списка кортежей, содержащих несколько значений индекса каждой точки:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
```

<!-- #region deletable=true editable=true -->
Его можно даже построить из декартова произведения отдельных индексов:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
```

<!-- #region deletable=true editable=true -->
Любой из этих объектов может быть передан как аргумент `index` при создании `Series` или `Dataframe`, или передан методу `reindex` существующего `Series` или `DataFrame`.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### Названия уровней мультииндексов

Иногда бывает удобно явно задавать названия уровней `MultiIndex`.
Это можно сделать, передав аргумент `names` любому из приведенных выше конструкторов `MultiIndex` или установив атрибут `names` индекса постфактум:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop.index.names = ['city', 'year']
pop
```

<!-- #region deletable=true editable=true -->
При наличии сложных наборов данных это может быть полезным способом отслеживания значения различных индексов.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### Мультииндекс для столбцов

В `DataFrame` строки и столбцы полностью симметричны, и так же, как строки могут иметь несколько уровней индексов, столбцы также могут иметь несколько уровней.
Рассмотрим следующий пример, представляющий собой имитацию некоторых медицинских данных:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
# иерархические индексы и столбцы
index = pd.MultiIndex.from_product([[2023, 2024], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Борис', 'Геннадий', 'Николай'], ['Pulse', 'Temp']],
                                     names=['subject', 'type'])

# Создаем имитационные данные
data = np.round(np.random.randn(4, 6), 1)
data[:, 1::2] += 37
data[:, ::2] += 70
data[:, ::2] //= 1

# Создаем  DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
```

<!-- #region deletable=true editable=true -->
Мультииндексация как строк, так и столбцов может оказаться чрезвычайно удобной. 
По сути дела, это четырехмерные данные со следующими измерениями: пациент, измеряемый параметр, год и номер посещения. 
При наличии этого можно, например, индексировать столбец верхнего уровня по имени человека и получить объект DataFrame, содержащий информацию только об этом человеке:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
health_data['Геннадий']
```

<!-- #region deletable=true editable=true -->
Для сложных записей, содержащих несколько маркированных неоднократно измеряемых параметров для многих субъектов (людей, стран, городов и т. д.), будет исключительно удобно использовать иерархические строки и столбцы!
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
## Индексация и срезы по MultiIndex

Индексация и срезы в `MultiIndex` разработаны так, чтобы быть интуитивно понятными, особенно если рассматривать индексы как дополнительные измерения.
Сначала рассмотрим индексацию многократно индексированных `Series`, а затем многократно индексированных `DataFrame`.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### Мультииндексация объектов Series

Рассмотрим мультииндексированный объект Series, содержащий сведения о населении городов:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop
```

<!-- #region deletable=true editable=true -->
Получить доступ к отдельным элементам можно, индексируя их с помощью нескольких индексов:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop['Новосибирск', 2022]
```

<!-- #region deletable=true editable=true -->
`MultiIndex` поддерживает и  *частичную индексацию* (*partial indexing*) или индексацию только одного из уровней индекса.
Результатом является еще один объект `Series` с сохраненными индексами более низкого уровня:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop['Новосибирск']
```

<!-- #region deletable=true editable=true -->
<!-- Возможно также выполнение частичных срезов, если мультииндекс отсортирован
(см. обсуждение в пункте «Отсортированные и неотсортированные индексы» подраздела «Перегруппировка мультииндексов» данного раздела:


pop.loc['Екатеринбург':'Красноярск']-->

Частичная индексация может быть выполнена по более низким уровням путем передачи пустого среза в первый индекс:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop[:, 2022]
```

<!-- #region deletable=true editable=true -->
Другие типы индексации и выбора  (обсуждаемые в [Индексация и выборка данных](pandas_02_data_indexing_and_selection.md)) также работают; например, выбор на основе булевых масок:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop[pop > 2200000]
```

<!-- #region deletable=true editable=true -->
Отбор на основе индексации списками также работает:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop[['Новосибирск', 'Казань']]
```

<!-- #region deletable=true editable=true -->
### Мультииндексация объектов `DataFrames`

Аналогичным образом ведет себя и мультииндексированные `DataFrame`.
Рассмотрим созданные ранее медицинские данные:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
health_data
```

<!-- #region deletable=true editable=true -->
Помните, что столбцы являются первичными в `DataFrame`, и синтаксис, используемый для мультииндексированных `Series`, применяется к столбцам.
Например, можно восстановить данные о частоте сердечных сокращений (пульсе) Геннадия с помощью простой операции:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
health_data['Геннадий', 'Pulse']
```

<!-- #region deletable=true editable=true -->
Также, как и в случае с одним индексом, можно использовать индексаторы `loc` и `iloc`, представленные в [Индексация и выборка данных](pandas_02_data_indexing_and_selection.md)). 
Например:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
health_data.iloc[:2, :2]
```

<!-- #region deletable=true editable=true -->
Эти индексаторы обеспечивают представление базовых двумерных данных в виде массива, но каждому отдельному индексу в `loc` или `iloc` может быть передан кортеж из нескольких индексов. 
Например:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
health_data.loc[:, ('Борис', 'Pulse')]
```

<!-- #region deletable=true editable=true -->
Работать со срезами в подобных кортежах индексов не очень удобно.
Например, попытка создать срез в кортеже может привести к синтаксической ошибке:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
%%capture

health_data.loc[(:, 1), (:, 'Pulse')]
```

<!-- #region deletable=true editable=true -->
Эту проблему можно обойти, явно построив нужный срез с помощью встроенной функции Python `slice()`, но в этом контексте лучшим способом будет использование объекта `IndexSlice`, который Pandas предоставляет именно для такой ситуации.
Например:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'Pulse']]
```

<!-- #region deletable=true editable=true -->
Существует множество способов взаимодействия с данными в мультииндексированных объектах `Series` и `DataFrame`, и лучший способ разобраться в них &mdash; начать с ними экспериментировать!
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
## Перегруппировка мультииндексов

Один из ключей к эффективной работе с мультииндексированными данными &mdash; умение эффективно преобразовывать данные.
Существует ряд операций, которые сохраняющих всю информацию в наборе данных, но переупорядочивающих его для удобства проведения различных вычислений.
Примером этому служат методы `stack()` и `unstack()`, но существует множество других способов точного управления перераспределением данных между иерархическими индексами и столбцами.
Рассмотрим их здесь.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
### Сортированные и несортированные индексы

*Большинство операций среза с мультииндексом не будут выполнены и завершатся ошибкой, если индекс не отсортирован!!!*

Начнем с создания простых многократно индексированных данных, где индексы *не отсортированы лексикографически*:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data
```

<!-- #region deletable=true editable=true -->
Попытка взять частичный срез этого индекса приведет к ошибке:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)
```

<!-- #region deletable=true editable=true -->
Хотя из сообщения об ошибке (&laquo;Длина ключа была больше, чем глубина лексикографической сортировки объекта Multi-
Index&raquo;) не совсем ясно, но это следствие того, что `MultiIndex` не отсортирован.
По разным причинам частичные срезы и другие подобные операции требуют, чтобы уровни в `MultiIndex` были в отсортированном (т. е. лексографическом) порядке.
Pandas предоставляет ряд удобных процедур для выполнения этого типа сортировки.
Примерами могут служить методы `sort_index()` и `sortlevel()` класса `DataFrame`.
Воспользуемся простейшим `sort_index()`:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
data = data.sort_index()
data
```

<!-- #region deletable=true editable=true -->
При такой сортировке индекса частичный срез будет работать так, как и ожидалось:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
data['a':'b']
```

<!-- #region deletable=true editable=true -->
### Выполнение над индексами операций `stack` и `unstack`

Метод `unstack` объекта `DataFrame` дает возможность преобразовывать набор данных из вертикального мультииндексированного в простое двумерное представление, при необходимости указывая требуемый уровень:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop.unstack(level=0)
```

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop.unstack(level=1)
```

<!-- #region deletable=true editable=true -->
Противоположностью `unstack()` является `stack()`, который можно использовать для восстановления исходного ряда данных:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop.unstack().stack()
```

<!-- #region deletable=true editable=true -->
### Настройка и сброс индекса

Другой способ реорганизации иерархических данных &mdash; преобразование меток индекса в столбцы, что можно сделать с помощью метода `reset_index`.
Вызов этого метода для набора данных о населении городов приведет к созданию `DataFrame` со столбцами *city* и *year*, содержащими информацию, которая ранее была в индексе.
Для большей ясности можно опционально указать название для представления столбцов данных:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop_flat = pop.reset_index(name='population')
pop_flat
```

<!-- #region deletable=true editable=true -->
Часто реальные необработанные входные данные выглядят именно таким образом, и полезно построить `MultiIndex` из значений столбцов.
Это можно сделать с помощью метода `set_index` объекта `DataFrame`, который возвращает мультииндексированный объект `DataFrame`:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
pop_flat.set_index(['city', 'year'])
```

<!-- #region deletable=true editable=true -->
На практике этот тип переиндексации является одним из наиболее полезных приемов при работе с реальными наборами данных.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
## Агрегирование данных по мультииндексам

Pandas имеет встроенные методы агрегации данных, такие как `mean()`, `sum()` и `max()`.
В случае иерархически индексированных данных им можно передать параметр `level` для указания подмножества данных, на котором будет вычисляться сводный показатель.

Например, вернемся к данным о здоровье:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
health_data
```

<!-- #region deletable=true editable=true -->
Пусть необходимо усреднить измерения в двух визитах каждый год. 
Можно сделать это, сгруппировав данные по индексу, который требуется исследовать, в данном случае год:
<!-- #endregion -->

```python deletable=true editable=true jupyter={"outputs_hidden": false}
data_mean =  health_data.groupby(level='year').mean()

data_mean
```

<!-- #region deletable=true editable=true -->
Подробнее функциональность `GroupBy`, будет рассмотрена в [Агрегирование и группировка](pandas_08_aggregation_and_grouping.md).
Хотя рассмотренный пример представляет собой лишь совмем маленький модельный набор данных, многие реальные наборы данных имеют схожую иерархическую структуру.
<!-- #endregion -->

<!-- #region deletable=true editable=true -->
## Объекты `Panel`

В Pandas есть еще несколько фундаментальных структур данных, которые, а именно объекты `pd.Panel` и `pd.Panel4D`.
Их можно рассматривать как трехмерное и четырехмерное обобщение (одномерной) структуры `Series` и (двумерной) структуры `DataFrame` соответственно.
Как только вы освоите индексацию и обработку данных в, 
Объекты `Panel` и `Panel4D` построены на тех же принципах, что и `Series` и `DataFrame`, поэтому манипулирование ими практически ничем не отличается от рассмотренного выше.
В частности, индексаторы `loc` и `iloc`, обсуждаемые в [Индексация и выборка данных](pandas_02_data_indexing_and_selection.md), предсказуемым образом работают с этмим многомерными структурами.

Поскольку, в большинстве случаев мультииндексация &mdash; это удобное и концептуально простое представление для многомерных данных, то многомерные структуры здесь рассматриваться не будут.
Помимо этого, многомерные данные по существу &mdash; это плотное представление данных, в то время как мультииндексация &mdash; разреженное представление данных. 
По мере увеличения размерности плотное представление становится все менее эффективным для большинства реальных наборов данных. 
Для получения дополнительной информации о структурах `Panel` и `Panel4D`, смотрите ссылки, перечисленные в [Дополнительные ресурсы](pandas_13_further_resources).
<!-- #endregion -->

```python

```
