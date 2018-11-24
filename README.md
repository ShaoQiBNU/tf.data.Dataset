tensorflow数据读取类——tf.data.Dataset
====================================

# 一. 背景

> 之前在TensorFlow中读取数据一般有两种方法：
>
> - 使用placeholder读内存中的数据
>
> - 使用queue读硬盘中的数据——https://zhuanlan.zhihu.com/p/27238630.

> Dataset API同时支持从内存和硬盘的读取，相比之前的两种方法在语法上更加简洁易懂。此外，如果想要用到TensorFlow新出的Eager模式，就必须要使用Dataset API来读取数据。

# 二. 数据集实例化

> Dataset是数据集生成器，可以看做是相同类型“元素”的有序列表，单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或dict；Iterator是迭代器，用于迭代对象实例化。

## 1. 一维数据集

```python
import tensorflow as tf
import numpy as np
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))
```

> 输出结果如下：

```python
1.0
2.0
3.0
4.0
5.0
```

> 如果一个dataset中元素被读取完了，再尝试sess.run(one_element)的话，就会抛出tf.errors.OutOfRangeError异常，这个行为与使用队列方式读取数据的行为是一致的。在实际程序中，可以在外界捕捉这个异常以判断数据是否读取完，代码如下：

```python
import tensorflow as tf
import numpy as np
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```
## 2. 高维数据集

> tf.data.Dataset.from_tensor_slices真正作用是切分传入Tensor的第一个维度，生成相应的dataset，即第一维表明数据集中数据的数量，之后切分batch等操作都以第一维为基础。例如：

```python
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
```

> 传入的数值是一个矩阵，它的形状为(5, 2)，tf.data.Dataset.from_tensor_slices就会切分它形状上的第一个维度，最后生成的dataset中一个含有5个元素，每个元素的形状是(2, )，即每个元素是矩阵的一行。

```python
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```

> 输出结果如下：

```python 
[0.4033448  0.52757601]
[0.28198747 0.29625652]
[0.57901578 0.58713477]
[0.84965332 0.49564784]
[0.23794543 0.99611239]
end!
```

## 3. 字典

> 在实际使用中，可能还希望Dataset中的每个元素具有更复杂的形式，如每个元素是一个Python中的元组，或是Python中的词典。例如，在图像识别问题中，一个元素可以是{“image”: image_tensor, “label”: label_tensor}的形式，这样处理起来更方便。注意，image_tensor、label_tensor和上面的高维向量一致，第一维表示数据集中数据的数量。相较之下，字典中每一个key值可以看做数据的一个属性，value则存储了所有数据的该属性值。代码如下：

```python
dataset = tf.data.Dataset.from_tensor_slices({
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                      
        "b": np.random.uniform(size=(5, 2))})
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```

> 输出结果如下：

```python
{'a': 1.0, 'b': array([0.92243988, 0.3671634 ])}
{'a': 2.0, 'b': array([0.79671281, 0.07432855])}
{'a': 3.0, 'b': array([0.61992803, 0.54869003])}
{'a': 4.0, 'b': array([0.52524662, 0.59375982])}
{'a': 5.0, 'b': array([0.92540254, 0.04185931])}
```

# 三. 数据集处理

> Dataset支持一类特殊的操作：Transformation。一个Dataset通过Transformation变成一个新的Dataset。通常我们可以通过Transformation完成数据变换，打乱，组成batch，生成epoch等一系列操作。常用的Transformation有：
>
> - map
>
> - batch
> - shuffle
> - repeat

## 1. map

> 和python中的map类似，map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset，代码如下：

```python
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.map(lambda x: x + 1)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```

> 输出结果如下：

```python
2.0
3.0
4.0
5.0
6.0
end!
```

## 2. batch

> batch就是将多个元素组合成batch，如上所说，按照输入元素第一个维度，代码如下：

```python
dataset = tf.data.Dataset.from_tensor_slices({
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                      
        "b": np.random.uniform(size=(5, 2))})
dataset = dataset.batch(2)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```

> 输出结果如下：

```python
{'a': array([1., 2.]), 'b': array([[0.27505219, 0.57905291],[0.69204933, 0.65733402]])}
{'a': array([3., 4.]), 'b': array([[0.21008234, 0.24788919],[0.94834079, 0.16925745]])}
{'a': array([5.]), 'b': array([[0.65639316, 0.59579377]])}
end!
```

## 3. shuffle

> shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小，建议舍的不要太小，一般是1000，代码如下：

```python
dataset = tf.data.Dataset.from_tensor_slices({
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                      
        "b": np.random.uniform(size=(5, 2))})
dataset = dataset.shuffle(buffer_size=5)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```

> 输出结果如下：

```python
{'a': 5.0, 'b': array([0.33428034, 0.96123072])}
{'a': 4.0, 'b': array([0.13703113, 0.71557694])}
{'a': 3.0, 'b': array([0.08029119, 0.76902701])}
{'a': 1.0, 'b': array([0.18158424, 0.08836979])}
{'a': 2.0, 'b': array([0.32355007, 0.63510833])}
```

## 4. repeat

> repeat的功能就是将整个序列重复多次，主要用来处理机器学习中的epoch，假设原先的数据是一个epoch，使用repeat(2)就可以将之变成2个epoch，代码如下：

```python
dataset = tf.data.Dataset.from_tensor_slices({
        "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                      
        "b": np.random.uniform(size=(5, 2))})
dataset = dataset.repeat(2)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
```

> 输出结果如下：

```python
{'a': 1.0, 'b': array([0.38191904, 0.59001444])}
{'a': 2.0, 'b': array([0.4665875 , 0.89079155])}
{'a': 3.0, 'b': array([0.02156276, 0.55409473])}
{'a': 4.0, 'b': array([0.63337527, 0.5299842 ])}
{'a': 5.0, 'b': array([0.3006349 , 0.93266542])}
{'a': 1.0, 'b': array([0.38191904, 0.59001444])}
{'a': 2.0, 'b': array([0.4665875 , 0.89079155])}
{'a': 3.0, 'b': array([0.02156276, 0.55409473])}
{'a': 4.0, 'b': array([0.63337527, 0.5299842 ])}
{'a': 5.0, 'b': array([0.3006349 , 0.93266542])}
end!
```

# 四. 实例

> 应用深层神经网络分类器DNNClassifier(Deep Neural Network Classifier)对鸢尾花数据进行分类。数据有两个文件：iris_training.csv和iris_test.csv，iris_training.csv是训练数据集，iris_test.csv是测试数据集。两个数据集的第一行是数据集的相关说明：

```python
iris_training.csv：120 4 setosa versicolor virginica
iris_test.csv：     30 4 setosa versicolor virginica
```

> 120/30代表行数；4代表每朵花的特征数，分别是花萼的长度、花萼的宽度、花瓣的长度、花瓣的宽度；setosa versicolor virginica是鸢尾花的三个类型：山鸢尾花Setosa、变色鸢尾花Versicolor、韦尔吉尼娅鸢尾花Virginica。下面的数字行代表数据集，每一行有5个数字，逗号分割，前面4个对应花萼花瓣4个特征，最后一个都是0、1或2，0代表Setosa，1代表Versicolor，2代表Virginica。代码如下：

```python
################# load packages #################
import pandas as pd
import tensorflow as tf

################# feature name and classes #################
FUTURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


################# load train data and test data #################
train_path='/Users/shaoqi/Desktop/iris_training.csv'
test_path='/Users/shaoqi/Desktop/iris_test.csv'

######### train data #########
train = pd.read_csv(train_path, names=FUTURES, header=0)
train_x, train_y = train, train.pop('Species')

######### test data ########
test = pd.read_csv(test_path, names=FUTURES, header=0)
test_x, test_y = test, test.pop('Species')


################# 设定特征值的名称 #################
feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


################# 选定估算器：深层神经网络分类器 #################
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3)


################# 针对训练的feed数据函数 #################
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)  # 每次随机调整数据顺序
    return dataset.make_one_shot_iterator().get_next()


################# 设定仅输出警告提示，可改为INFO #################
tf.logging.set_verbosity(tf.logging.WARN)


################# 训练模型 #################
batch_size = 100
classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=1000)


################# 针对测试的feed数据函数 #################
def eval_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


################# 评估 #################
eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, batch_size))
print(eval_result)


################# 支持100次循环对新数据进行分类预测 #################
for i in range(0, 100):
    print('\nPlease enter features: SepalLength,SepalWidth,PetalLength,PetalWidth')
    a, b, c, d = map(float, input().split(','))  # 捕获用户输入的数字
    predict_x = {
        'SepalLength': [a],
        'SepalWidth': [b],
        'PetalLength': [c],
        'PetalWidth': [d],
    }

    ########## 预测 #########
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x,
                                       labels=[0],
                                       batch_size=batch_size))

    ########## 预测结果及可能性 #########
    for pred_dict in predictions:
        print(pred_dict)
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(SPECIES[class_id], 100 * probability)
```

> 输出结果：

```python
{'accuracy': 0.93333334, 'average_loss': 0.06302313, 'loss': 1.8906939, 'global_step': 1000}

Please enter features: SepalLength,SepalWidth,PetalLength,PetalWidth
    
2.3,4.5,1.5,3

{'logits': array([-2.37141  ,  1.0964978, -1.2131451], dtype=float32), 'probabilities': array([0.02758318, 0.8845809 , 0.08783597], dtype=float32), 'class_ids': array([1]), 'classes': array([b'1'], dtype=object)}

Versicolor 88.4580910205841
```
