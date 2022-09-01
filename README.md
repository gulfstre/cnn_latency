# Предсказание времени выполнения cnn модели на основе её признаков

Работа проведена в <a href="https://colab.research.google.com/drive/1KWTFaDezHJ04t6w_AP5ckLgfiDdk9yMX?usp=sharing">этом colab</a>

Задача поставлена достаточно абстрактно, поэтому, по крайней мере, в первых подходах добавим некоторой частности. Будем рассматривать реальную рабочую архитектуру, в качестве таковой возьмём ResNet.


## Генерация моделей

<img src=".github/cd3cf4f38d.jpeg" width="900" style="max-width: 100%;">

Будем варьировать основные параметры в рамках архитектуры: 
  - тип используемых блоков (выделены красным)
  - количество блоков в каждом layer (conv2_x, conv3_x, conv4_x, conv5_x)
  - количество фильтров

Количество блоков в каждом layer будем варьировать от =1 до 4-кратного максимального значения среди стандартных архитектур (те что на рис.). Количество фильтров будет задавать одним числом. В стандартных архитектурах это всегда 64 (на старте первого layer, а остальные пропорциональны). Мы будем варьировать 
этот параметр от =1 до 3-кратного стандартного. 


## Генерация датасета

Будем прогонять каждую модель на cpu colab по одному на тензорах полученных из нескольких случайно (с постоянным random seed) выбранных картинок из имеющейся коллекции и фиксировать среднее время. В качестве параметров будем использовать те, что варьировали при генерации моделей.

## Датасеты

Сгенерированы 2 варианта датасета: с постоянным количеством фильтров и с вариацией по их количеству.


## Предсказание времени выполнения

Попробуем несколько простых моделей: LinearRegression, DecisionTreeRegressor, RandomForestRegressor.


## Результаты и дальнейшие исследования
Мы получили весьма хорошие результаты по r2_score на простых моделях в зависимости от основных параметров. Обобщение принятых нами допущений об архитектуре, использование более детальной параметризации, более сложных регрессоров и прочего в рамках данной постановки задачи не видится целесообразным. Для дальнейшего исследования следует понимать более конкретную цель.
