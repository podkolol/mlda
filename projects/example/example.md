---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Проект. Прогнозирование цен на недвижимость


## **Цель:**  
Разработать модель машинного обучения для прогнозирования цен на недвижимость на основе различных факторов, таких как местоположение, площадь, количество комнат и другие характеристики.


## **Актуальность:**  
Прогнозирование цен на недвижимость важно для рынка жилья, риелторов, покупателей и инвесторов. Это помогает принимать обоснованные решения о покупке, продаже или аренде недвижимости. Автоматизация этого процесса может значительно упростить работу аналитиков и повысить точность оценки.


## **Набор данных:**  
[Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
Этот набор данных содержит информацию о домах в одном из городов США, включая их характеристики (площадь, количество комнат, год постройки и т.д.) и цены.


## **Шаги выполнения:**

1. **Изучение данных**  
   Загрузите данные и проведите предварительный анализ:
   - Проверьте наличие пропущенных значений.
   - Визуализируйте распределение целевой переменной (цены).
   - Исследуйте корреляцию между признаками и ценой.

2. **Предобработка данных**  
   - Обработайте пропущенные значения (например, заполните средними или медианами).
   - Преобразуйте категориальные признаки в числовые с помощью One-Hot Encoding или Label Encoding.
   - Нормализуйте числовые признаки для улучшения работы модели.

3. **Создание модели**  
   - Разделите данные на обучающую и тестовую выборки (например, 80/20).
   - Используйте линейную регрессию, случайный лес (Random Forest) или нейронные сети для прогнозирования.
   - Настройте гиперпараметры модели (например, глубину деревьев в случайном лесе).

4. **Обучение и оценка модели**  
   - Обучите модель на обучающей выборке.
   - Оцените её производительность на тестовой выборке с использованием метрик RMSE (Root Mean Squared Error) и MAE (Mean Absolute Error).

5. **Визуализация результатов**  
   - Создайте график реальных и предсказанных цен.
   - Проанализируйте ошибки модели и предложите способы их уменьшения.

6. **Деплой модели**  
   - Интегрируйте модель в простое веб-приложение (например, с использованием Flask или Streamlit), чтобы пользователи могли вводить параметры дома и получать прогнозируемую цену.


## **Пример выполнения:**

**Шаг 1: Загрузка и исследование данных**  

```python
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Загрузка данных
data = pd.read_csv("./data/train.csv")
data = data[['GrLivArea', 'BedroomAbvGr', 'YearBuilt', 'SalePrice']]
```

```python
# Первые строки датасета
print(data.head())
```

```python
# Статистика по числам
print(data.describe())
```

```python
# Визуализация распределения цен
plt.hist(data['SalePrice'], bins=30)
plt.title("Распределение цен на недвижимость")
plt.xlabel("Цена")
plt.ylabel("Количество домов")
plt.show()
```

**Шаг 2: Предобработка данных**  

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Разделение на признаки и целевую переменную
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Категориальные и числовые признаки
categorical_features = X.select_dtypes(include="object").columns
numerical_features = X.select_dtypes(exclude="object").columns

# Создание пайплайна для предобработки
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_features),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Шаг 3: Создание и обучение модели**  

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Создание пайплайна с моделью
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse}, MAE: {mae}")
```

**Шаг 4: Визуализация результатов**  

```python
import numpy as np

# График реальных и предсказанных цен
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
plt.title("Реальные vs Предсказанные цены")
plt.xlabel("Реальная цена")
plt.ylabel("Предсказанная цена")
plt.show()

# Анализ ошибок
errors = y_test - y_pred
plt.hist(errors, bins=30)
plt.title("Распределение ошибок")
plt.xlabel("Ошибка")
plt.ylabel("Количество")
plt.show()
```

###### **Шаг 5: Сохранение модели**  
Сохраните модель в файл для последующего использования:

```python
import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
```

###### **Шаг 6: Деплой модели**  
Создайте простое веб-приложение с использованием Flask или Streamlit:

```python
%%writefile app.py

import streamlit as st
import pickle
import pandas as pd


# Загрузка обученной модели
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Интерфейс приложения
st.title("Прогнозирование цен на недвижимость")

# Ввод параметров
area = st.number_input("Площадь дома (в кв. футах)", min_value=500, max_value=10000)
rooms = st.number_input("Количество комнат", min_value=1, max_value=10)
year_built = st.number_input("Год постройки", min_value=1800, max_value=2023)

# Предсказание
if st.button("Предсказать"):
    input_data = pd.DataFrame({
        "GrLivArea": [area],
        "BedroomAbvGr": [rooms],
        "YearBuilt": [year_built]
    })
    prediction = model.predict(input_data)
    st.success(f"Прогнозируемая цена: ${prediction[0]:,.2f}")
```

###### **Шаг 7: Запуск модели**  

```bash

# Чтоб не запрашивал почту
mkdir -p ./.streamlit
echo '[general]\nemail = "example@mail.mail"' > ./.streamlit/credentials.toml

# Запуск модели в "продакшене"
streamlit run app.py
```
