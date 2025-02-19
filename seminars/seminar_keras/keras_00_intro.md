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

<!-- #region id="RNubmTU69Hlm" editable=true slideshow={"slide_type": ""} -->
# Основы Keras
<!-- #endregion -->

<!-- #region id="xXBaDmqO_KB1" editable=true slideshow={"slide_type": ""} -->
Keras — это мощная и удобная в использовании библиотека для глубокого обучения, которая позволяет строить и тренировать сложные модели нейронных сетей с минимальным усилием. Она разработана с акцентом на удобство пользователя, модульность и расширяемость. Вот несколько ключевых аспектов и возможностей Keras:

### Простота и Интуитивность
Keras обладает простым и интуитивно понятным API, который делает процесс создания моделей глубокого обучения более доступным и менее трудоемким, даже для начинающих пользователей. С Keras можно легко экспериментировать с различными архитектурами нейронных сетей.

### Модульность
Keras построен как набор настраиваемых модулей, которые могут быть связаны вместе с небольшими ограничениями. Это означает, что нейронные слои, функции потерь, оптимизаторы, инициализаторы, активационные функции и регуляризаторы могут быть комбинированы различными способами для соответствия конкретным задачам.

### Поддержка различных бэкендов
Keras может использовать несколько фреймворков глубокого обучения как бэкенд, включая TensorFlow, Microsoft Cognitive Toolkit (CNTK), и Theano. Самым популярным и часто используемым бэкендом является TensorFlow. Это означает, что пользователи могут выбирать между этими бэкендами, исходя из своих предпочтений и требований к производительности.

### Поддержка множества типов сетей
Keras поддерживает широкий спектр сетей, включая полносвязные сети, сверточные нейронные сети (CNN), рекуррентные нейронные сети (RNN), а также комбинации этих типов для более сложных архитектур. Keras также поддерживает работу с временными последовательностями и текстами благодаря встроенным функциям для обработки и генерации последовательностей.

### Широкая экосистема
Keras имеет большое и активное сообщество пользователей и разработчиков, благодаря чему постоянно появляются новые учебные ресурсы, готовые к использованию модели и расширения. Существует множество предварительно обученных моделей, доступных для задач, таких как классификация изображений, генерация текста и многие другие.

### Интеграция с TensorFlow 2.0
С выпуском TensorFlow 2.0, Keras был тесно интегрирован в TensorFlow как его официальный высокоуровневый API, что сделало его еще более мощным и удобным инструментом для разработки моделей глубокого обучения.
<!-- #endregion -->

<!-- #region id="5om3jJPjAT6-" editable=true slideshow={"slide_type": ""} -->
## Общий алгоритм формирования модели ИНС
<!-- #endregion -->

<!-- #region id="gk9b7jC89LN5" editable=true slideshow={"slide_type": ""} -->
Создание моделей искусственных нейронных сетей в Keras можно свести к нескольким основным шагам:

### 1. Установка и настройка

Убедитесь, что у вас установлены Python и Keras. Keras обычно идёт в комплекте с TensorFlow, который можно установить через pip:

```
pip install tensorflow
```

### 2. Импорт необходимых модулей

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
```

### 3. Подготовка данных

Подготовьте ваши данные. Для обучения модели данные должны быть разделены на два или три набора: обучающий, проверочный и, опционально, тестовый. Данные также должны быть предобработаны (нормализация, аугментация и т. д.).

### 4. Создание модели

Используйте модель `Sequential` для построения модели слой за слоем. Например, для простой полносвязной сети:

```python
model = Sequential([
  Dense(128, activation='relu', input_shape=(FEATURES,)),
  Dropout(0.2),
  Dense(NUM_CLASSES, activation='softmax')
])
```

Для задачи классификации изображений с CNN:

```python
model = Sequential([
  Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)),
  MaxPooling2D(pool_size=(2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(NUM_CLASSES, activation='softmax')
])
```

### 5. Компиляция модели

Выберите оптимизатор, функцию потерь и метрики для оценки. Затем скомпилируйте модель:

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 6. Обучение модели

Обучите модель, используя ваши обучающие данные. Вы также можете передать проверочные данные, чтобы отслеживать производительность модели во время обучения.

```python
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

### 7. Оценка и использование модели

После обучения оцените модель на тестовом наборе данных и используйте её для предсказаний.

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nТестовая точность:', test_acc)

predictions = model.predict(x_test)
```

### 8. Сохранение и загрузка модели

Сохраните модель для последующего использования и загрузите её при необходимости.

```python
model.save('my_model.h5')  # Сохранить модель
del model  # Удалить текущую модель
model = keras.models.load_model('my_model.h5')  # Загрузить модель
```
<!-- #endregion -->

<!-- #region id="xiGG2A_PCHRR" editable=true slideshow={"slide_type": ""} -->
## Подробное описание основных процессов и дополнительные функции
<!-- #endregion -->

<!-- #region id="XLJX2YpaAMWF" editable=true slideshow={"slide_type": ""} -->
### Полносвязные Слои (Dense Layers)

Полносвязные слои, или Dense слои в Keras, являются основой для построения нейронных сетей, предназначенных для решения широкого спектра задач. Каждый нейрон в полносвязном слое соединен со всеми нейронами предыдущего уровня, что позволяет обучать модель на выявление сложных зависимостей в данных.

Создание такого слоя в Keras выглядит следующим образом:

```python
from tensorflow.keras.layers import Dense

# Создаем полносвязный слой с 128 нейронами и функцией активации ReLU
dense_layer = Dense(128, activation='relu')
```

### Сборка Модели

Для создания модели в Keras обычно используется Sequential API, который позволяет последовательно добавлять слои:

```python
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),  # Первый скрытый слой
    Dense(num_classes, activation='softmax')  # Выходной слой для классификации
])
```

`input_shape` обозначает размерность входных данных, а `num_classes` - количество классов для задачи классификации.

### Решение Задач Регрессии

Для задач регрессии выходной слой модели обычно содержит один нейрон (для одномерного выхода) с линейной функцией активации (или без нее), что позволяет предсказывать непрерывные значения:

```python
model.add(Dense(1, activation='linear'))
```

В качестве функции потерь обычно используется среднеквадратичная ошибка (`mean_squared_error`).

### Решение Задач Классификации

Для задач классификации выходной слой должен содержать количество нейронов, равное числу классов, и использовать функцию активации `softmax` для многоклассовой классификации или `sigmoid` для бинарной:

```python
model.add(Dense(num_classes, activation='softmax'))  # Для многоклассовой классификации
```

Функция потерь для многоклассовой классификации чаще всего — `categorical_crossentropy`, а для бинарной — `binary_crossentropy`.

### Компиляция Модели

После создания архитектуры модели необходимо скомпилировать ее, указав оптимизатор, функцию потерь и метрики для оценки производительности:

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Обучение Модели

Обучение модели в Keras производится с помощью метода `fit`, где указываются обучающие данные, количество эпох и размер мини-пакета (batch size):

```python
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

Создание слоев и моделей в Keras демонстрирует баланс между простотой использования и гибкостью настройки для улучшения производительности модели и расширения возможностей Keras.

### Валидация во время обучения

Чтобы контролировать переобучение модели в процессе обучения, можно использовать валидационный набор данных. Это позволяет оценить производительность модели на данных, которые не использовались в процессе обучения:

```python
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
```

Валидационные данные могут быть использованы для ранней остановки обучения или для настройки гиперпараметров модели.


### Использование Callbacks

Callbacks в Keras - это набор функций, которые могут быть применены в различных точках процесса обучения (например, в конце эпохи). Они могут использоваться для различных целей, включая сохранение модели в процессе обучения, изменение скорости обучения, раннюю остановку и т.д.:

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Callback для сохранения модели после каждой эпохи
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', save_best_only=True, monitor='val_loss', mode='min')

# Callback для ранней остановки
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

callbacks_list = [checkpoint, early_stopping]

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32, callbacks=callbacks_list)
```

### Настройка Оптимизаторов

Keras предоставляет широкий спектр оптимизаторов, таких как SGD, Adam, RMSprop и другие. Каждый из них может быть настроен с помощью различных параметров (скорость обучения, момент и т.д.):

```python
from tensorflow.keras.optimizers import Adam

# Настройка оптимизатора Adam
optimizer = Adam(lr=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

### Пользовательские Функции Потерь и Метрики

В случае, если стандартных функций потерь или метрик недостаточно для решения задачи, можно определить собственные функции:

```python
import tensorflow as tf

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
```
### Автоматическое настройка гиперпараметров

Автоматическая настройка гиперпараметров может значительно улучшить производительность моделей, минимизируя усилия, связанные с ручным поиском оптимальных значений. В экосистеме TensorFlow для этой цели часто используется Keras Tuner. Этот инструмент позволяет автоматически оптимизировать гиперпараметры модели, такие как скорость обучения, архитектура сети и другие:

```python
import kerastuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=[input_shape]))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

tuner.search(x_train, y_train, epochs=50, validation_split=0.2)
```

После завершения поиска можно получить наилучшую модель и ее гиперпараметры:

```python
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]
```

### Использование предварительно обученных моделей

В Keras доступно множество предварительно обученных моделей, таких как VGG, Inception и ResNet, которые можно использовать как для предсказания, так и для трансферного обучения. Трансферное обучение позволяет адаптировать заранее обученную модель под новую задачу с минимальным количеством данных:

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Замораживаем слои базовой модели
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Многозадачное обучение

В Keras можно легко реализовать многозадачное обучение, когда одна модель обучается одновременно на нескольких задачах. Это может быть полезно для решения схожих задач, где модель может извлечь общие признаки для нескольких выходных данных:

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_tensor = Input(shape=(input_shape,))
shared_layers = Dense(128, activation='relu')(input_tensor)

output_1 = Dense(num_classes_1, activation='softmax')(shared_layers)
output_2 = Dense(num_classes_2, activation='softmax')(shared_layers)

model = Model(inputs=input_tensor, outputs=[output_1, output_2])

model.compile(optimizer='adam',
              loss=['categorical_crossentropy', 'categorical_crossentropy'],
              metrics=['accuracy'])
```

### Улучшенное обучение с помощью аугментации данных

Аугментация данных является мощным способом улучшения производительности моделей глубокого обучения, особенно когда доступно ограниченное количество данных. В Keras это можно реализовать с помощью `ImageDataGenerator`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Создаем генератор аугментации данных для тренировочного набора
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Генератор для валидационного набора данных (без аугментации)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Применяем аугментацию к тренировочным изображениям
train_generator = train_datagen.flow_from_directory(
        train_directory,  # путь к директории с тренировочными изображениями
        target_size=(150, 150),  # все изображения будут изменены до 150x150
        batch_size=32,
        class_mode='binary')  # для бинарной классификации используется class_mode='binary'

validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
```

Для обучения модели с использованием этих генераторов:

```python
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # количество итераций за эпоху
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)  # количество итераций для валидации за эпоху
```

### Регуляризация для уменьшения переобучения

Регуляризация помогает уменьшить переобучение путем добавления штрафов на веса модели. В Keras это можно сделать, добавив слои `Dropout` или используя регуляризаторы весов:

```python
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(1024, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_shape,)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
<!-- #endregion -->

<!-- #region id="68CEc5MC-cB3" -->
# **11. Ускорение обучения моделей при помощи GPU**
<!-- #endregion -->

<!-- #region id="tMce8muBqXQP" -->
Этот блок представляет собой введение в вычисления на [GPU](https://cloud.google.com/gpu) в Colab. В этом блоке вы подключитесь к GPU, а затем выполните несколько базовых операций TensorFlow как на CPU, так и на GPU, наблюдая за ускорением, которое дает использование GPU.
<!-- #endregion -->

<!-- #region id="oM_8ELnJq_wd" -->
## Включение и тестирование GPU

Сначала вам нужно включить GPU для ноутбука:

- Перейдите в меню Среда выполнения→Сменить среду выполнения
- выберите GPU из выпадающего списка.

Далее мы убедимся, что можем подключиться к GPU с помощью tensorflow:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="sXnDmXR7RDr2" outputId="02deb19d-e110-42cb-ffbd-495e79a5be16" editable=true slideshow={"slide_type": ""}
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

<!-- #region id="v3fE7KmKRDsH" editable=true slideshow={"slide_type": ""} -->
## Наблюдаем ускорение TensorFlow на GPU по сравнению с CPU

В этом примере строится сверточной слой нейронной сети на основе
случайно сгенерированного изображения, вначале используя CPU, а затем GPU, чтобы сравнить скорость выполнения.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Y04m-jvKRDsJ" outputId="f01fd9e4-2d0d-44bb-8381-3dabbbb531c2" editable=true slideshow={"slide_type": ""}
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nЭта ошибка, скорее всего, означает, что этот ноутбук не '
      'настроен на использование графического процессора.  Измените это в настройках ноутбука с помощью '
      'командную палитру (cmd/ctrl-shift-P) или меню Edit.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)

# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->

<!-- #endregion -->
