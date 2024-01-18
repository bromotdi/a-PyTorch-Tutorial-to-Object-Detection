Это **[PyTorch](https://pytorch.org) руководство по детекции объектов**.

Это третье из [серии руководств](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch). Я пишу о самостоятельной _реализации_ классных моделей с помощью потрясающей библиотеки PyTorch.

Предполагается базовое знание PyTorch и сверточных нейронных сетей.

Если вы новичок в PyTorch, сначала прочитайте [Глубокое обучение с PyTorch: 60-минутный блиц](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) и [Изучение PyTorch на примерах](https:// pytorch.org/tutorials/beginner/pytorch_with_examples.html).

Вопросы, предложения или исправления можно публиковать как issues.

Я использую PyTorch 0.4 в Python 3.6.

---

**4 ноября 2023 г.**: 中文翻译 – китайский перевод этого руководства был любезно предоставлен пользователем [@zigerZZZ](https://github.com/zigerZZZ) – см. [README_zh.md](https:// github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/README_zh.md).

---

# Содержание

[***Цель***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#objective)

[***Концепции***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#concepts)

[***Обзор***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#overview)

[***Реализация***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#implementation)

[***Обучение***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#training)

[***Оценка***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#evaluation)

[***Вывод***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#inference)

[***Часто задаваемые вопросы***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#faqs)

# Цель

**Построить модель, которая сможет определять местоположение конкретных объектов на изображениях.**

<p выравнивание="центр">
<img src="./img/baseball.gif">
</p>

Для этой задачи мы будем внедрять [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325), популярную, мощную и особенно гибкую сеть. Оригинальную реализацию авторов можно найти [здесь](https://github.com/weiliu89/caffe/tree/ssd).

Вот несколько примеров детекции объектов на изображениях, которые не использовались во время обучения:

---

<p align="center">
<img src="./img/000001.jpg">
</p>

---

<p align="center">
<img src="./img/000022.jpg">
</p>

---

<p align="center">
<img src="./img/000069.jpg">
</p>

---

<p align="center">
<img src="./img/000082.jpg">
</p>

---

<p align="center">
<img src="./img/000144.jpg">
</p>

---

<p align="center">
<img src="./img/000139.jpg">
</p>

---

<p align="center">
<img src="./img/000116.jpg">
</p>

---

<p align="center">
<img src="./img/000098.jpg">
</p>

---

Дополнительные примеры можно найти в [конце руководства] (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#some-more-examples).

---

# Концепции

* **Обнаружение объектов**.

* **Single-Shot Detection (SSD)**. Более ранние архитектуры детекции объектов состояли из двух отдельных этапов: модуля Region Proposal Network (RPN), выполняющего локализацию объектов, и классификатора для определения классов объектов в предлагаемых регионах (proposed regions). С вычислительной точки зрения это может быть очень затратно и поэтому плохо подходит для приложений, работающих в режиме реального времени. Single-shot models объединяют задачи локализации и классификации в одном проходе по сети, что приводит к более быстрой детекции при развертывании на более легком оборудовании.

* **Multiscale Feature Maps (Многомасштабные карты признаков)**. В задачах классификации изображений мы делаем предсказания на основе последней карте признаков после свертки – наименьшего, но наиболее глубокого представления исходного изображения. В детекции объектов карты признаков с промежуточных сверточных слоев также могут быть _непосредственно_ полезны, поскольку они представляют исходное изображение в разных масштабах. Следовательно, фильтр фиксированного размера, работающий на разных картах признаков, сможет обнаружить объекты различных размеров.

* **Priors**. Это заранее рассчитанные боксы, заданные в определенных местах на конкретных картах признаков, с определенными соотношениями сторон и масштабами. Они тщательно выбираются таким образом, чтобы соответствовать характеристикам ограничивающих боксов объектов (т.е. ground truths) в наборе данных.

* **Multibox**. Это [метод] (https://arxiv.org/abs/1312.2249), который формулирует прогнозирование ограничивающей рамки объекта как  _задачу регрессии_, при которой координаты обнаруженного объекта регрессируются к его истинных координатам. Кроме того, для каждого предсказанного бокса генерируются оценки (scores) для различных классов объектов. Priors служат возможными отправными точками для предсказаний, поскольку они моделируются на основе ground truths. Следовательно, количество предсказанных рамок будет равно количеству priors, большинство из которых не содержат объекта.

* **Hard Negative Mining**. Это относится к выбору наиболее явных ложно-положительных срабатываний (False Positive), предсказанных моделью, и принуждению ее учиться на этих примерах. Другими словами, мы анализируем только те ложно-положительные срабатывания, которые модели _труднее_ всего идентифицировать правильно. В контексте детекции объектов, где подавляющее большинство предсказанных боксов не содержит объекта, это также способствует уменьшению дисбаланса между числом ложно-положительных и истинно-положительных срабатываний.

* **Non-Maximum Suppression**. На любом заданном месте несколько priors могут существенно перекрываться. Поэтому предсказания, возникающие из этих priors, могут фактически являться дубликатами одного и того же объекта. Метод Non-Maximum Suppression (NMS) представляет собой способ удаления избыточных предсказаний, подавляя все, кроме того, у которого максимальная оценка.

# Обзор

В этом разделе я представлю обзор данной модели. Если вы уже знакомы с ним, вы можете сразу перейти к разделу [Реализация](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#implementation) или к коду с комментариями.

По мере того, как мы продолжим, вы заметите, что в структуре и формулировке SSD заложено немало инженерных решений. Не волнуйтесь, если некоторые аспекты поначалу покажутся надуманными или не слишком спонтанными. Помните, что они основаны на _многолетних_ исследованиях (часто эмпирических) в данной области.

### Некоторые определения

Box - это просто бокс. _bounding_ box - это ограничивающий бокс, который охватывает объект и представляет его границы.

В этом руководстве мы столкнемся с обеими типами - обычными боксами и ограничивающими боксами. Но для всех боксов, представленных на изображениях, нам нужно уметь измерять их положение, форму, размеры и другие свойства.

#### Граничные координаты

Самый очевидный способ представить бокс — это использовать пиксельные координаты линий `x` и `y`, которые составляют его границы.

![](./img/bc1.PNG)

Граничные координаты бокса — это просто **`(x_min, y_min, x_max, y_max)`**.

Но значения пикселей практически бесполезны, если мы не знаем фактических размеров изображения.
Лучшим способом было бы представить все координаты в их дробной форме.

![](./img/bc2.PNG)

Теперь координаты не зависят от размера, и все боксы на всех изображениях измеряются в одном масштабе.

####  Center-Size coordinates (Координаты центра и размера)

Это более явный способ представления положения и размеров бокса.

![](./img/cs.PNG)

Координаты центра и размера бокса представляют собой: **`(c_x, c_y, w, h)`**.

В коде вы обнаружите, что мы обычно используем обе системы координат в зависимости от их пригодности для задачи, но _всегда_ в их дробных формах.

#### Jaccard Index (Коэффициент Жаккара)

Мера Жаккара или Jaccard Overlap или метрика Intersection over Union (IoU) измеряют **меру перекрытия двух боксов**.

![](./img/jaccard.jpg)

Значение IoU, равное `1`, означает, что это один и тот же бокс, а значение `0` указывает, что это взаимоисключающие области.

Это простая метрика, но она находит множество применений в нашей модели.

### Multibox

Multibox — это метод для детекции объектов, где предсказание состоит из двух компонентов:

- **Координаты бокса, который может содержать или не содержать объект**. Это задача _регрессии_.

- **Оценки для различных классов объектов для этого бокса**, включая класс _background_ (фона), который подразумевает, что в боксе нет объектов. Это задача _классификации_.

### Single Shot Detector (SSD)

SSD — это полностью сверточная нейронная сеть (CNN), которую мы можем разделить на три части:

- __Базовые свертки__, полученные из существующей архитектуры классификации изображений, которые предоставят карты признаков более низкого уровня.

- __Вспомогательные свертки__ (__Auxiliary convolutions__), добавленные поверх базовой сети, которые предоставят карты признаков более высокого уровня.

- __Прогнозирующие свертки__, которые будут находить и идентифицировать объекты на этих картах признаков.

В статье демонстрируются два варианта модели под названием SSD300 и SSD512. Суффиксы представляют размер входного изображения. Хотя эти две сети немного отличаются по способу построения, в принципе они одинаковы. SSD512 — это просто более крупная сеть, обеспечивающая немного лучшую производительность.

Для удобства мы рассмотрим SSD300.

### Базовые свертки – 1 часть

Прежде всего, зачем использовать свертки из существующей архитектуры нейронной сети?

Потому что модели, доказавшие свою эффективность в классификации изображений, уже довольно хорошо улавливают основные черты изображения. Те же самые сверточные слои полезны для детекции объектов, хотя и в более _локальном_ смысле: нас интересуют конкретные области изображения, где присутствуют объекты, а не изображение в целом.

Дополнительным преимуществом является возможность использовать слои, предварительно обученные на основе надежного набора данных для классификации. Как вы, возможно, знаете, это называется **Трансферное обучение**. Заимствуя знания из другой, но тесно связанной задачи, мы добились прогресса, даже не начав.

Авторы статьи используют **архитектуру VGG-16** в качестве базовой модели. В ее первоначальной форме она довольно проста.

![](./img/vgg16.PNG)

Они рекомендуют использовать модель, предварительно обученную на задаче классификации _ImageNet Large Scale Visual Recognition Competition (ILSVRC)_. К счастью, такая модель уже доступна в PyTorch, также как и другие популярные архитектуры. Если хотите, вы можете выбрать что-то побольше, например ResNet. Просто помните о вычислительных требованиях.

Согласно статье, **мы должны внести некоторые изменения в эту предварительно обученную сеть**, чтобы адаптировать ее к нашей собственной задаче детекции объектов. Некоторые изменения логичны и необходимы, в то время как другие скорее вопрос удобства или предпочтения.

– **Размер входного изображения** будет равен `300, 300`, как было указано ранее.

- **3-й слой пулинга**, который уменьшает размеры вдвое, будет использовать математическую функцию `ceiling` вместо используемой по умолчанию функции `floor` для определения размера выходного слоя. Это важно только в том случае, если размеры предыдущей карты признаков являются нечетные, а не четные. Глядя на изображение выше, вы можете подсчитать, что для входного изображения размером `300, 300` карта признаков `conv3_3` будет иметь поперечное сечение `75, 75`, которое уменьшит размеры вдвое до `38, 38`, вместо неудобного `37, 37`.

- Мы модифицируем **5-й слой пулинга** с ядром `2, 2` и шагом `2` на ядро `3, 3` и шаг `1`. Это означает, что теперь размеры карты признаков из предыдущего сверточного слоя уже не уменьшатся вдвое.

- Нам не нужны полносвязные (т.е. классификационные) слои, потому что они не несут никакой ценной информации для данной задачи. Мы полностью отбрасываем `fc8`, но решаем **_переделать_ `fc6` и `fc7` в сверточные слои `conv6` и `conv7`**.

Первые три модификации достаточно просты, но последняя, вероятно, требует некоторых объяснений.

### FC → Convolutional Layer

How do we reparameterize a fully connected layer into a convolutional layer?

Consider the following scenario.

In the typical image classification setting, the first fully connected layer cannot operate on the preceding feature map or image _directly_. We'd need to flatten it into a 1D structure.

![](./img/fcconv1.jpg)

In this example, there's an image of dimensions `2, 2, 3`, flattened to a 1D vector of size `12`. For an output of size `2`, the fully connected layer computes two dot-products of this flattened image with two vectors of the same size `12`. **These two vectors, shown in gray, are the parameters of the fully connected layer.**

Now, consider a different scenario where we use a convolutional layer to produce `2` output values.

![](./img/fcconv2.jpg)

Here, the image of dimensions `2, 2, 3` need not be flattened, obviously. The convolutional layer uses two filters with `12` elements in the same shape as the image to perform two dot products. **These two filters, shown in gray, are the parameters of the convolutional layer.**

But here's the key part – **in both scenarios, the outputs `Y_0` and `Y_1` are the same!**

![](./img/fcconv3.jpg)

The two scenarios are equivalent.

What does this tell us?

That **on an image of size `H, W` with `I` input channels, a fully connected layer of output size `N` is equivalent to a convolutional layer with kernel size equal to the image size `H, W` and `N` output channels**, provided that the parameters of the fully connected network `N, H * W * I` are the same as the parameters of the convolutional layer `N, H, W, I`.

![](./img/fcconv4.jpg)

Therefore, any fully connected layer can be converted to an equivalent convolutional layer simply **by reshaping its parameters**.

### Base Convolutions – part 2

We now know how to convert `fc6` and `fc7` in the original VGG-16 architecture into `conv6` and `conv7` respectively.

In the ImageNet VGG-16 [shown previously](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#base-convolutions--part-1), which operates on images of size `224, 224, 3`, you can see that the output of `conv5_3` will be of size `7, 7, 512`. Therefore –

- `fc6` with a flattened input size of `7 * 7 * 512` and an output size of `4096` has parameters of dimensions `4096, 7 * 7 * 512`. **The equivalent convolutional layer `conv6` has a `7, 7` kernel size and `4096` output channels, with reshaped parameters of dimensions `4096, 7, 7, 512`.**

- `fc7` with an input size of `4096` (i.e. the output size of `fc6`) and an output size `4096` has parameters of dimensions `4096, 4096`. The input could be considered as a `1, 1` image with `4096` input channels. **The equivalent convolutional layer `conv7` has a `1, 1` kernel size and `4096` output channels, with reshaped parameters of dimensions `4096, 1, 1, 4096`.**

We can see that `conv6` has `4096` filters, each with dimensions `7, 7, 512`, and `conv7` has `4096` filters, each with dimensions `1, 1, 4096`.

These filters are numerous and large – and computationally expensive.

To remedy this, the authors opt to **reduce both their number and the size of each filter by subsampling parameters** from the converted convolutional layers.

- `conv6` will use `1024` filters, each with dimensions `3, 3, 512`. Therefore, the parameters are subsampled from `4096, 7, 7, 512` to `1024, 3, 3, 512`.

- `conv7` will use `1024` filters, each with dimensions `1, 1, 1024`. Therefore, the parameters are subsampled from `4096, 1, 1, 4096` to `1024, 1, 1, 1024`.

Based on the references in the paper, we will **subsample by picking every `m`th parameter along a particular dimension**, in a process known as [_decimation_](https://en.wikipedia.org/wiki/Downsampling_(signal_processing)).  

Since the kernel of `conv6` is decimated from `7, 7` to `3,  3` by keeping only every 3rd value, there are now _holes_ in the kernel. Therefore, we would need to **make the kernel dilated or _atrous_**.

This corresponds to a dilation of `3` (same as the decimation factor `m = 3`). However, the authors actually use a dilation of `6`, possibly because the 5th pooling layer no longer halves the dimensions of the preceding feature map.

We are now in a position to present our base network, **the modified VGG-16**.

![](./img/modifiedvgg.PNG)

In the above figure, pay special attention to the outputs of `conv4_3` and `conv_7`. You will see why soon enough.

### Auxiliary Convolutions

We will now **stack some more convolutional layers on top of our base network**. These convolutions provide additional feature maps, each progressively smaller than the last.

![](./img/auxconv.jpg)

We introduce four convolutional blocks, each with two layers. While size reduction happened through pooling in the base network, here it is facilitated by a stride of `2` in every second layer.

Again, take note of the feature maps from `conv8_2`, `conv9_2`, `conv10_2`, and `conv11_2`.

### A detour

Before we move on to the prediction convolutions, we must first understand what it is we are predicting. Sure, it's objects and their positions, _but in what form?_

It is here that we must learn about _priors_ and the crucial role they play in the SSD.

#### Priors

Object predictions can be quite diverse, and I don't just mean their type. They can occur at any position, with any size and shape. Mind you, we shouldn't go as far as to say there are _infinite_ possibilities for where and how an object can occur. While this may be true mathematically, many options are simply improbable or uninteresting. Furthermore, we needn't insist that boxes are pixel-perfect.

In effect, we can discretize the mathematical space of potential predictions into just _thousands_ of possibilities.

**Priors are precalculated, fixed boxes which collectively represent this universe of probable and approximate box predictions**.

Priors are manually but carefully chosen based on the shapes and sizes of ground truth objects in our dataset. By placing these priors at every possible location in a feature map, we also account for variety in position.

In defining the priors, the authors specify that –

- **they will be applied to various low-level and high-level feature maps**, viz. those from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2`, and `conv11_2`. These are the same feature maps indicated on the figures before.

- **if a prior has a scale `s`, then its area is equal to that of a square with side `s`**. The largest feature map, `conv4_3`, will have priors with a scale of `0.1`, i.e. `10%` of image's dimensions, while the rest have priors with scales linearly increasing from `0.2` to `0.9`. As you can see, larger feature maps have priors with smaller scales and are therefore ideal for detecting smaller objects.

- **At _each_ position on a feature map, there will be priors of various aspect ratios**. All feature maps will have priors with ratios `1:1, 2:1, 1:2`. The intermediate feature maps of `conv7`, `conv8_2`, and `conv9_2` will _also_ have priors with ratios `3:1, 1:3`. Moreover, all feature maps will have *one extra prior* with an aspect ratio of `1:1` and at a scale that is the geometric mean of the scales of the current and subsequent feature map.

| Feature Map From | Feature Map Dimensions | Prior Scale | Aspect Ratios | Number of Priors per Position | Total Number of Priors on this Feature Map |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| `conv4_3`      | 38, 38       | 0.1 | 1:1, 2:1, 1:2 + an extra prior | 4 | 5776 |
| `conv7`      | 19, 19       | 0.2 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 2166 |
| `conv8_2`      | 10, 10       | 0.375 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 600 |
| `conv9_2`      | 5, 5       | 0.55 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 150 |
| `conv10_2`      | 3,  3       | 0.725 | 1:1, 2:1, 1:2 + an extra prior | 4 | 36 |
| `conv11_2`      | 1, 1       | 0.9 | 1:1, 2:1, 1:2 + an extra prior | 4 | 4 |
| **Grand Total**      |    –    | – | – | – | **8732 priors** |

There are a total of 8732 priors defined for the SSD300!

#### Visualizing Priors

We defined the priors in terms of their _scales_ and _aspect ratios_.

![](./img/wh1.jpg)

Solving these equations yields a prior's dimensions `w` and `h`.

![](./img/wh2.jpg)

We're now in a position to draw them on their respective feature maps.

For example, let's try to visualize what the priors will look like at the central tile of the feature map from `conv9_2`.

![](./img/priors1.jpg)

The same priors also exist for each of the other tiles.

![](./img/priors2.jpg)

#### Predictions vis-à-vis Priors

[Earlier](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#multibox), we said we would use regression to find the coordinates of an object's bounding box. But then, surely, the priors can't represent our final predicted boxes?

They don't.

Again, I would like to reiterate that the priors represent, _approximately_, the possibilities for prediction.

This means that **we use each prior as an approximate starting point and then find out how much it needs to be adjusted to obtain a more exact prediction for a bounding box**.

So if each predicted bounding box is a slight deviation from a prior, and our goal is to calculate this deviation, we need a way to measure or quantify it.

Consider a cat, its predicted bounding box, and the prior with which the prediction was made.  

![](./img/ecs1.PNG)

Assume they are represented in center-size coordinates, which we are familiar with.

Then –

![](./img/ecs2.PNG)

This answers the question we posed at the [beginning of this section](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#a-detour). Considering that each prior is adjusted to obtain a more precise prediction, **these four offsets `(g_c_x, g_c_y, g_w, g_h)` are the form in which we will regress bounding boxes' coordinates**.

As you can see, each offset is normalized by the corresponding dimension of the prior. This makes sense because a certain offset would be less significant for a larger prior than it would be for a smaller prior.

### Prediction convolutions

Earlier, we earmarked and defined priors for six feature maps of various scales and granularity, viz. those from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2`, and `conv11_2`.

Then, **for _each_ prior at _each_ location on _each_ feature map**, we want to predict –

- the **offsets `(g_c_x, g_c_y, g_w, g_h)`** for a bounding box.

- a set of **`n_classes` scores** for the bounding box, where `n_classes` represents the total number of object types (including a _background_ class).

To do this in the simplest manner possible, **we need two convolutional layers for each feature map** –

- a **_localization_ prediction** convolutional layer with a `3,  3` kernel evaluating at each location (i.e. with padding and stride of `1`) with `4` filters for _each_ prior present at the location.

  The `4` filters for a prior calculate the four encoded offsets `(g_c_x, g_c_y, g_w, g_h)` for the bounding box predicted from that prior.

- a **_class_ prediction** convolutional layer with a `3,  3` kernel evaluating at each location (i.e. with padding and stride of `1`) with `n_classes` filters for _each_ prior present at the location.

  The `n_classes` filters for a prior calculate a set of `n_classes` scores for that prior.

![](./img/predconv1.jpg)

All our filters are applied with a kernel size of `3, 3`.

We don't really need kernels (or filters) in the same shapes as the priors because the different filters will _learn_ to make predictions with respect to the different prior shapes.

Let's take a look at the **outputs of these convolutions**. Consider again the feature map from `conv9_2`.

![](./img/predconv2.jpg)

The outputs of the localization and class prediction layers are shown in blue and yellow respectively. You can see that the cross-section (`5, 5`) remains unchanged.

What we're really interested in is the _third_ dimension, i.e. the channels. These contain the actual predictions.

If you **choose a tile, _any_ tile, in the localization predictions and expand it**, what will you see?

![](./img/predconv3.jpg)

Voilà! The channel values at each position of the localization predictions represent the encoded offsets with respect to the priors at that position.

Now, **do the same with the class predictions.** Assume `n_classes = 3`.

![](./img/predconv4.jpg)

Similar to before, these channels represent the class scores for the priors at that position.

Now that we understand what the predictions for the feature map from `conv9_2` look like, we can **reshape them into a more amenable form.**

![](./img/reshaping1.jpg)

We have arranged the `150` predictions serially. To the human mind, this should appear more intuitive.

But let's not stop here. We could do the same for the predictions for _all_ layers and stack them together.

We calculated earlier that there are a total of 8732 priors defined for our model. Therefore, there will be **8732 predicted boxes in encoded-offset form, and 8732 sets of class scores**.

![](./img/reshaping2.jpg)

**This is the final output of the prediction stage.** A stack of boxes, if you will, and estimates for what's in them.

It's all coming together, isn't it? If this is your first rodeo in object detection, I should think there's now a faint light at the end of the tunnel.

### Multibox loss

Based on the nature of our predictions, it's easy to see why we might need a unique loss function. Many of us have calculated losses in regression or classification settings before, but rarely, if ever, _together_.

Obviously, our total loss must be an **aggregate of losses from both types of predictions** – bounding box localizations and class scores.

Then, there are a few questions to be answered –

>_What loss function will be used for the regressed bounding boxes?_

>_Will we use multiclass cross-entropy for the class scores?_

>_In what ratio will we combine them?_

>_How do we match predicted boxes to their ground truths?_

>_We have 8732 predictions! Won't most of these contain no object? Do we even consider them?_

Phew. Let's get to work.

#### Matching predictions to ground truths

Remember, the nub of any supervised learning algorithm is that **we need to be able to match predictions to their ground truths**. This is tricky since object detection is more open-ended than the average learning task.

For the model to learn _anything_, we'd need to structure the problem in a way that allows for comparisons between our predictions and the objects actually present in the image.

Priors enable us to do exactly this!

- **Find the Jaccard overlaps** between the 8732 priors and `N` ground truth objects. This will be a tensor of size `8732, N`.

- **Match** each of the 8732 priors to the object with which it has the greatest overlap.

- If a prior is matched with an object with a **Jaccard overlap of less than `0.5`**, then it cannot be said to "contain" the object, and is therefore a **_negative_ match**. Considering we have thousands of priors, most priors will test negative for an object.

- On the other hand, a handful of priors will actually **overlap significantly (greater than `0.5`)** with an object, and can be said to "contain" that object. These are **_positive_ matches**.

- Now that we have **matched each of the 8732 priors to a ground truth**, we have, in effect, also **matched the corresponding 8732 predictions to a ground truth**.  

Let's reproduce this logic with an example.

![](./img/matching1.PNG)

For convenience, we will assume there are just seven priors, shown in red. The ground truths are in yellow – there are three actual objects in this image.

Following the steps outlined earlier will yield the following matches –

![](./img/matching2.jpg)

Now, **each prior has a match**, positive or negative. By extension, **each prediction has a match**, positive or negative.

Predictions that are positively matched with an object now have ground truth coordinates that will serve as **targets for localization**, i.e. in the _regression_ task. Naturally, there will be no target coordinates for negative matches.

All predictions have a ground truth label, which is either the type of object if it is a positive match or a _background_ class if it is a negative match. These are used as **targets for class prediction**, i.e. the _classification_ task.

#### Localization loss

We have **no ground truth coordinates for the negative matches**. This makes perfect sense. Why train the model to draw boxes around empty space?

Therefore, the localization loss is computed only on how accurately we regress positively matched predicted boxes to the corresponding ground truth coordinates.

Since we predicted localization boxes in the form of offsets `(g_c_x, g_c_y, g_w, g_h)`, we would also need to encode the ground truth coordinates accordingly before we calculate the loss.

The localization loss is the averaged **Smooth L1** loss between the encoded offsets of positively matched localization boxes and their ground truths.

![](./img/locloss.jpg)

#### Confidence loss

Every prediction, no matter positive or negative, has a ground truth label associated with it. It is important that the model recognizes both objects and a lack of them.

However, considering that there are usually only a handful of objects in an image, **the vast majority of the thousands of predictions we made do not actually contain an object**. As Walter White would say, _tread lightly_. If the negative matches overwhelm the positive ones, we will end up with a model that is less likely to detect objects because, more often than not, it is taught to detect the _background_ class.

The solution may be obvious – limit the number of negative matches that will be evaluated in the loss function. But how do we choose?

Well, why not use the ones that the model was most _wrong_ about? In other words, only use those predictions where the model found it hardest to recognize that there are no objects. This is called **Hard Negative Mining**.

The number of hard negatives we will use, say `N_hn`, is usually a fixed multiple of the number of positive matches for this image. In this particular case, the authors have decided to use three times as many hard negatives, i.e. `N_hn = 3 * N_p`. The hardest negatives are discovered by finding the Cross Entropy loss for each negatively matched prediction and choosing those with top `N_hn` losses.

Then, the confidence loss is simply the sum of the **Cross Entropy** losses among the positive and hard negative matches.

![](./img/confloss.jpg)

You will notice that it is averaged by the number of positive matches.

#### Total loss

The **Multibox loss is the aggregate of the two losses**, combined in a ratio `α`.

![](./img/totalloss.jpg)

In general, we needn't decide on a value for `α`. It could be a learnable parameter.

For the SSD, however, the authors simply use `α = 1`, i.e. add the two losses. We'll take it!

### Processing predictions

After the model is trained, we can apply it to images. However, the predictions are still in their raw form – two tensors containing the offsets and class scores for 8732 priors. These would need to be processed to **obtain final, human-interpretable bounding boxes with labels.**

This entails the following –

- We have 8732 predicted boxes represented as offsets `(g_c_x, g_c_y, g_w, g_h)` from their respective priors. Decode them to boundary coordinates, which are actually directly interpretable.

- Then, for each _non-background_ class,

  - Extract the scores for this class for each of the 8732 boxes.

  - Eliminate boxes that do not meet a certain threshold for this score.

  - The remaining (uneliminated) boxes are candidates for this particular class of object.

At this point, if you were to draw these candidate boxes on the original image, you'd see **many highly overlapping boxes that are obviously redundant**. This is because it's extremely likely that, from the thousands of priors at our disposal, more than one prediction corresponds to the same object.

For instance, consider the image below.

![](./img/nms1.PNG)

There's clearly only three objects in it – two dogs and a cat. But according to the model, there are _three_ dogs and _two_ cats.

Mind you, this is just a mild example. It could really be much, much worse.

Now, to you, it may be obvious which boxes are referring to the same object. This is because your mind can process that certain boxes coincide significantly with each other and a specific object.

In practice, how would this be done?

First, **line up the candidates for each class in terms of how _likely_ they are**.

![](./img/nms2.PNG)

We've sorted them by their scores.

The next step is to find which candidates are redundant. We already have a tool at our disposal to judge how much two boxes have in common with each other – the Jaccard overlap.

So, if we were to **draw up the Jaccard similarities between all the candidates in a given class**, we could evaluate each pair and **if found to overlap significantly, keep only the _more likely_ candidate**.

![](./img/nms3.jpg)

Thus, we've eliminated the rogue candidates – one of each animal.

This process is called __Non-Maximum Suppression (NMS)__ because when multiple candidates are found to overlap significantly with each other such that they could be referencing the same object, **we suppress all but the one with the maximum score**.

Algorithmically, it is carried out as follows –

- Upon selecting candidates for each _non-background_ class,

  - Arrange candidates for this class in order of decreasing likelihood.

  - Consider the candidate with the highest score. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than, say, `0.5` with this candidate.

  - Consider the next highest-scoring candidate still remaining in the pool. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than `0.5` with this candidate.

  - Repeat until you run through the entire sequence of candidates.

The end result is that you will have just a single box – the very best one – for each object in the image.

![](./img/nms4.PNG)

Non-Maximum Suppression is quite crucial for obtaining quality detections.

Happily, it's also the final step.

# Implementation

The sections below briefly describe the implementation.

They are meant to provide some context, but **details are best understood directly from the code**, which is quite heavily commented.

### Dataset

We will use Pascal Visual Object Classes (VOC) data from the years 2007 and 2012.

#### Description

This data contains images with twenty different types of objects.

```python
{'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
```

Each image can contain one or more ground truth objects.

Each object is represented by –

- a bounding box in absolute boundary coordinates

- a label (one of the object types mentioned above)

-  a perceived detection difficulty (either `0`, meaning _not difficult_, or `1`, meaning _difficult_)

#### Download

Specifically, you will need to download the following VOC datasets –

- [2007 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (460MB)

- [2012 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB)

- [2007 _test_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (451MB)

Consistent with the paper, the two _trainval_ datasets are to be used for training, while the VOC 2007 _test_ will serve as our test data.  

Make sure you extract both the VOC 2007 _trainval_ and 2007 _test_ data to the same location, i.e. merge them.

### Inputs to model

We will need three inputs.

#### Images

Since we're using the SSD300 variant, the images would need to be sized at `300, 300` pixels and in the RGB format.

Remember, we're using a VGG-16 base pretrained on ImageNet that is already available in PyTorch's `torchvision` module. [This page](https://pytorch.org/docs/master/torchvision/models.html) details the preprocessing or transformation we would need to perform in order to use this model – pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.

Therefore, **images fed to the model must be a `Float` tensor of dimensions `N, 3, 300, 300`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.

#### Objects' Bounding Boxes

We would need to supply, for each image, the bounding boxes of the ground truth objects present in it in fractional boundary coordinates `(x_min, y_min, x_max, y_max)`.

Since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the bounding boxes for the entire batch of `N` images.

Therefore, **ground truth bounding boxes fed to the model must be a list of length `N`, where each element of the list is a `Float` tensor of dimensions `N_o, 4`**, where `N_o` is the number of objects present in that particular image.

#### Objects' Labels

We would need to supply, for each image, the labels of the ground truth objects present in it.

Each label would need to be encoded as an integer from `1` to `20` representing the twenty different object types. In addition, we will add a _background_ class with index `0`, which indicates the absence of an object in a bounding box. (But naturally, this label will not actually be used for any of the ground truth objects in the dataset.)

Again, since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the labels for the entire batch of `N` images.

Therefore, **ground truth labels fed to the model must be a list of length `N`, where each element of the list is a `Long` tensor of dimensions `N_o`**, where `N_o` is the number of objects present in that particular image.

### Data pipeline

As you know, our data is divided into _training_ and _test_ splits.

#### Parse raw data

See `create_data_lists()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This parses the data downloaded and saves the following files –

- A **JSON file for each split with a list of the absolute filepaths of `I` images**, where `I` is the total number of images in the split.

- A **JSON file for each split with a list of `I` dictionaries containing ground truth objects, i.e. bounding boxes in absolute boundary coordinates, their encoded labels, and perceived detection difficulties**. The `i`th dictionary in this list will contain the objects present in the `i`th image in the previous JSON file.

- A **JSON file which contains the `label_map`**, the label-to-index dictionary with which the labels are encoded in the previous JSON file. This dictionary is also available in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) and directly importable.

#### PyTorch Dataset

See `PascalVOCDataset` in [`datasets.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py).

This is a subclass of PyTorch [`Dataset`](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset), used to **define our training and test datasets.** It needs a `__len__` method defined, which returns the size of the dataset, and a `__getitem__` method which returns the `i`th image, bounding boxes of the objects in this image, and labels for the objects in this image, using the JSON files we saved earlier.

You will notice that it also returns the perceived detection difficulties of each of these objects, but these are not actually used in training the model. They are required only in the [Evaluation](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#evaluation) stage for computing the Mean Average Precision (mAP) metric. We also have the option of filtering out _difficult_ objects entirely from our data to speed up training at the cost of some accuracy.

Additionally, inside this class, **each image and the objects in them are subject to a slew of transformations** as described in the paper and outlined below.

#### Data Transforms

See `transform()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This function applies the following transformations to the images and the objects in them –

- Randomly **adjust brightness, contrast, saturation, and hue**, each with a 50% chance and in random order.

- With a 50% chance, **perform a _zoom out_ operation** on the image. This helps with learning to detect small objects. The zoomed out image must be between `1` and `4` times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.

- Randomly crop image, i.e. **perform a _zoom in_ operation.** This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between `0.3` and `1` times the original dimensions. The aspect ratio is to be between `0.5` and `2`. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either `0`, `0.1`, `0.3`, `0.5`, `0.7`, or `0.9`, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.

- With a 50% chance, **horizontally flip** the image.

- **Resize** the image to `300, 300` pixels. This is a requirement of the SSD300.

- Convert all boxes from **absolute to fractional boundary coordinates.** At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.

- **Normalize** the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.

As mentioned in the paper, these transformations play a crucial role in obtaining the stated results.

#### PyTorch DataLoader

The `Dataset` described above, `PascalVOCDataset`, will be used by a PyTorch [`DataLoader`](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) in `train.py` to **create and feed batches of data to the model** for training or evaluation.

Since the number of objects vary across different images, their bounding boxes, labels, and difficulties cannot simply be stacked together in the batch. There would be no way of knowing which objects belong to which image.

Instead, we need to **pass a collating function to the `collate_fn` argument**, which instructs the `DataLoader` about how it should combine these varying size tensors. The simplest option would be to use Python lists.

### Base Convolutions

See `VGGBase` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply base convolutions.**

The layers are initialized with parameters from a pretrained VGG-16 with the `load_pretrained_layers()` method.

We're especially interested in the lower-level feature maps that result from `conv4_3` and `conv7`, which we return for use in subsequent stages.

### Auxiliary Convolutions

See `AuxiliaryConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply auxiliary convolutions.**

Use a [uniform Xavier initialization](https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_uniform_) for the parameters of these layers.

We're especially interested in the higher-level feature maps that result from `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, which we return for use in subsequent stages.

### Prediction Convolutions

See `PredictionConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply localization and class prediction convolutions** to the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`.

These layers are initialized in a manner similar to the auxiliary convolutions.

We also **reshape the resulting prediction maps and stack them** as discussed. Note that reshaping in PyTorch is only possible if the original tensor is stored in a [contiguous](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous) chunk of memory.

As expected, the stacked localization and class predictions will be of dimensions `8732, 4` and `8732, 21` respectively.

### Putting it all together

See `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, the **base, auxiliary, and prediction convolutions are combined** to form the SSD.

There is a small detail here – the lowest level features, i.e. those from `conv4_3`, are expected to be on a significantly different numerical scale compared to its higher-level counterparts. Therefore, the authors recommend L2-normalizing and then rescaling _each_ of its channels by a learnable value.

### Priors

See `create_prior_boxes()` under `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

This function **creates the priors in center-size coordinates** as defined for the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, _in that order_. Furthermore, for each feature map, we create the priors at each tile by traversing it row-wise.

This ordering of the 8732 priors thus obtained is very important because it needs to match the order of the stacked predictions.

### Multibox Loss

See `MultiBoxLoss` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Two empty tensors are created to store localization and class prediction targets, i.e. _ground truths_, for the 8732 predicted boxes in each image.

We **find the ground truth object with the maximum Jaccard overlap for each prior**, which is stored in `object_for_each_prior`.

We want to avoid the rare situation where not all of the ground truth objects have been matched. Therefore, we also **find the prior with the maximum overlap for each ground truth object**, stored in `prior_for_each_object`. We explicitly add these matches to `object_for_each_prior` and artificially set their overlaps to a value above the threshold so they are not eliminated.

Based on the matches in `object_for_each prior`, we set the corresponding labels, i.e. **targets for class prediction**, to each of the 8732 priors. For those priors that don't overlap significantly with their matched objects, the label is set to _background_.

Also, we encode the coordinates of the 8732 matched objects in `object_for_each prior` in offset form `(g_c_x, g_c_y, g_w, g_h)` with respect to these priors, to form the **targets for localization**. Not all of these 8732 localization targets are meaningful. As we discussed earlier, only the predictions arising from the non-background priors will be regressed to their targets.

The **localization loss** is the [Smooth L1 loss](https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss) over the positive matches.

Perform Hard Negative Mining – rank class predictions matched to _background_, i.e. negative matches, by their individual [Cross Entropy losses](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss). The **confidence loss** is the Cross Entropy loss over the positive matches and the hardest negative matches. Nevertheless, it is averaged only by the number of positive matches.

The **Multibox Loss is the aggregate of these two losses**, combined in the ratio `α`. In our case, they are simply being added because `α = 1`.

# Training

Before you begin, make sure to save the required data files for training and evaluation. To do this, run the contents of [`create_data_lists.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/create_data_lists.py) after pointing it to the `VOC2007` and `VOC2012` folders in your [downloaded data](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#download).

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py).

The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you need to.

To **train your model from scratch**, run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.

### Remarks

In the paper, they recommend using **Stochastic Gradient Descent** in batches of `32` images, with an initial learning rate of `1e−3`, momentum of `0.9`, and `5e-4` weight decay.

I ended up using a batch size of `8` images for increased stability. If you find that your gradients are exploding, you could reduce the batch size, like I did, or clip gradients.

The authors also doubled the learning rate for bias parameters. As you can see in the code, this is easy do in PyTorch, by passing [separate groups of parameters](https://pytorch.org/docs/stable/optim.html#per-parameter-options) to the `params` argument of its [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD).

The paper recommends training for 80000 iterations at the initial learning rate. Then, it is decayed by 90% (i.e. to a tenth) for an additional 20000 iterations, _twice_. With the paper's batch size of `32`, this means that the learning rate is decayed by 90% once after the 154th epoch and once more after the 193th epoch, and training is stopped after 232 epochs. I followed this schedule.

On a TitanX (Pascal), each epoch of training required about 6 minutes.

I should note here that two unintended differences from the paper were brought to my attention by readers of this tutorial:

- My priors that overshoot the edges of the image are not being clipped, as pointed out in issue [#94](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/94) by _@AakiraOtok_. This does not appear to have a negative effect on performance, however, as discussed in that issue and also verified in issue [#95](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/95) by the same reader. It is even possible that there is a slight improvement in performance, but this may be too small to be conclusive.

- I mistakenly used L1 loss instead of *smooth* L1 loss as the localization loss, as pointed out in issue [#60](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/60) by _jonathan016_. This also appears to have no negative effect on performance as pointed out in that issue, but _smooth_ L1 loss may offer better training stability with larger batch sizes as mentioned in [this comment](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/94#issuecomment-1590217018). 

### Model checkpoint

You can download this pretrained model [here](https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load) for evaluation or inference – see below.

# Evaluation

See [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py).

The data-loading and checkpoint parameters for evaluating the model are at the beginning of the file, so you can easily check or modify them should you wish to.

To begin evaluation, simply run the `evaluate()` function with the data-loader and model checkpoint. **Raw predictions for each image in the test set are obtained and parsed** with the checkpoint's `detect_objects()` method, which implements [this process](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#processing-predictions). Evaluation has to be done at a `min_score` of `0.01`, an NMS `max_overlap` of `0.45`, and `top_k` of `200` to allow fair comparision of results with the paper and other implementations.

**Parsed predictions are evaluated against the ground truth objects.** The evaluation metric is the _Mean Average Precision (mAP)_. If you're not familiar with this metric, [here's a great explanation](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

We will use `calculate_mAP()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) for this purpose. As is the norm, we will ignore _difficult_ detections in the mAP calculation. But nevertheless, it is important to include them from the evaluation dataset because if the model does detect an object that is considered to be _difficult_, it must not be counted as a false positive.

The model scores **77.2 mAP**, same as the result reported in the paper.

Class-wise average precisions (not scaled to 100) are listed below.

| Class | Average Precision |
| :-----: | :------: |
| _aeroplane_ | 0.7887580990791321 |
| _bicycle_ | 0.8351995348930359 |
| _bird_ | 0.7623348236083984 |
| _boat_ | 0.7218425273895264 |
| _bottle_ | 0.45978495478630066 |
| _bus_ | 0.8705356121063232 |
| _car_ | 0.8655831217765808 |
| _cat_ | 0.8828985095024109 |
| _chair_ | 0.5917483568191528 |
| _cow_ | 0.8255912661552429 |
| _diningtable_ | 0.756867527961731 |
| _dog_ | 0.856262743473053 |
| _horse_ | 0.8778411149978638 |
| _motorbike_ | 0.8316892385482788 |
| _person_ | 0.7884440422058105 |
| _pottedplant_ | 0.5071538090705872 |
| _sheep_ | 0.7936667799949646 |
| _sofa_ | 0.7998116612434387 |
| _train_ | 0.8655905723571777 |
| _tvmonitor_ | 0.7492395043373108 |

You can see that some objects, like bottles and potted plants, are considerably harder to detect than others.

# Inference

See [`detect.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py).

Point to the model you want to use for inference with the `checkpoint` parameter at the beginning of the code.

Then, you can use the `detect()` function to identify and visualize objects in an RGB image.

```python
img_path = '/path/to/ima.ge'
original_image = PIL.Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
```

This function first **preprocesses the image by resizing and normalizing its RGB channels** as required by the model. It then **obtains raw predictions from the model, which are parsed** by the `detect_objects()` method in the model. The parsed results are converted from fractional to absolute boundary coordinates, their labels are decoded with the `label_map`, and they are **visualized on the image**.

There are no one-size-fits-all values for `min_score`, `max_overlap`, and `top_k`. You may need to experiment a little to find what works best for your target data.

### Some more examples

---

<p align="center">
<img src="./img/000029.jpg">
</p>

---

<p align="center">
<img src="./img/000045.jpg">
</p>

---

<p align="center">
<img src="./img/000062.jpg">
</p>

---

<p align="center">
<img src="./img/000075.jpg">
</p>

---

<p align="center">
<img src="./img/000085.jpg">
</p>

---

<p align="center">
<img src="./img/000092.jpg">
</p>

---

<p align="center">
<img src="./img/000100.jpg">
</p>

---

<p align="center">
<img src="./img/000124.jpg">
</p>

---

<p align="center">
<img src="./img/000127.jpg">
</p>

---

<p align="center">
<img src="./img/000128.jpg">
</p>

---

<p align="center">
<img src="./img/000145.jpg">
</p>

---

# FAQs

__I noticed that priors often overshoot the `3, 3` kernel employed in the prediction convolutions. How can the kernel detect a bound (of an object) outside it?__

Don't confuse the kernel and its _receptive field_, which is the area of the original image that is represented in the kernel's field-of-view.

For example, on the `38, 38` feature map from `conv4_3`, a `3, 3` kernel covers an area of `0.08, 0.08` in fractional coordinates. The priors are `0.1, 0.1`, `0.14, 0.07`, `0.07, 0.14`, and `0.14, 0.14`.

But its receptive field, which [you can calculate](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807), is a whopping `0.36, 0.36`! Therefore, all priors (and objects contained therein) are present well inside it.

Keep in mind that the receptive field grows with every successive convolution. For `conv_7` and the higher-level feature maps, a `3, 3` kernel's receptive field will cover the _entire_ `300, 300` image. But, as always, the pixels in the original image that are closer to the center of the kernel have greater representation, so it is still _local_ in a sense.

---

__While training, why can't we match predicted boxes directly to their ground truths?__

We cannot directly check for overlap or coincidence between predicted boxes and ground truth objects to match them because predicted boxes are not to be considered reliable, _especially_ during the training process. This is the very reason we are trying to evaluate them in the first place!

And this is why priors are especially useful. We can match a predicted box to a ground truth box by means of the prior it is supposed to be approximating. It no longer matters how correct or wildly wrong the prediction is.

---

__Why do we even have a _background_ class if we're only checking which _non-background_ classes meet the threshold?__

When there is no object in the approximate field of the prior, a high score for _background_ will dilute the scores of the other classes such that they will not meet the detection threshold.

---

__Why not simply choose the class with the highest score instead of using a threshold?__

I think that's a valid strategy. After all, we implicitly conditioned the model to choose _one_ class when we trained it with the Cross Entropy loss. But you will find that you won't achieve the same performance as you would with a threshold.

I suspect this is because object detection is open-ended enough that there's room for doubt in the trained model as to what's really in the field of the prior. For example, the score for _background_ may be high if there is an appreciable amount of backdrop visible in an object's bounding box. There may even be multiple objects present in the same approximate region. A simple threshold will yield all possibilities for our consideration, and it just works better.

Redundant detections aren't really a problem since we're NMS-ing the hell out of 'em.


---

__Sorry, but I gotta ask... _[what's in the boooox?!](https://cnet4.cbsistatic.com/img/cLD5YVGT9pFqx61TuMtcSBtDPyY=/570x0/2017/01/14/6d8103f7-a52d-46de-98d0-56d0e9d79804/se7en.png)___

Ha.
