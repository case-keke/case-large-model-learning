---
tags:
  - 深度学习
writer: 李显
---
# 数据预处理
### 记录原因

本节由于网上系统性、针对性（网上好像都是pytorch教学）资料缺乏，对于我这样的初学者困难较多；为了能节省一些其他同学的时间，同时也方便自己记忆，将本节学习内容、相关扩展知识按顺序整理如下

（有gpt的用起来，以下内容很多也整理自gpt）
### 数据下载

[MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges](http://yann.lecun.com/exdb/mnist/?ref=floydhub.ghost.io)
解压缩后，尝试运行以下代码：
```Python
import os

data_dir = 'mnist_path'  #此处填写文件路径，可在属性中查看
files = ['train-images-idx3-ubyte.gz', 
         'train-labels-idx1-ubyte.gz',
         't10k-images-idx3-ubyte.gz', 
         't10k-labels-idx1-ubyte.gz']

for file in files:
    file_path = os.path.join(data_dir, file)  #路径拼接
    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
    else:
        print(f'File found: {file_path}')
```

os.path.join 和 os.path.exist 是os库中常用的两个路径操作函数，值得记忆
**需要注意：**由于转义字符影响，**需要将路径中部分单斜杠改为双斜杠** 或者 **在路径前加上r**

### 数据读取

为了读取数据，需要用到以下函数：

#### gzip.open

（“数据下载”中的数据是以gz压缩文件形式存在，需要解压缩）
**`gzip.open(filename, mode='rb', compresslevel=9)`**
- 打开一个 Gzip 文件进行读写。
    
- **参数**:
    
    - `filename`: 文件名（字符串）。
        
    - `mode`: 打开模式（如 `'rb'` 读取，`'wb'` 写入）。
        
    - `compresslevel`: 压缩级别（0-9），默认是 9（最高压缩）。**（本节无需使用）**
        
#### file.read

读取 file 的全部内容，并返回一个字符串（对于T文本文件）或字节串（对于二进制文件）
#### numpy.frombuffer

**`numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)`**

- 参数
    
    - **`buffer`**:
        
        - 要读取的缓冲区（字节对象）。可以是 `bytes` 或 `bytearray`。
            
    - **`dtype`**:
        
        - 数据类型，指明要创建的数组的数据类型（如 `np.float32`、`np.int32` 等）。默认为 `float`。
            
    - **`count`**:
        
        - 可选，指定要读取的元素数量。默认为 `-1`，表示读取整个缓冲区。
            
    - **`offset`**:
        
        - 可选，指明从缓冲区的哪个字节开始读取。默认为 `0`。
            
- 返回值
    
    - 返回一个 NumPy 数组，数组的形状由 `buffer` 的字节数、`dtype` 和 `count` 决定。
        
offset的功能需要记忆，**因为无论是图像文件还是标签文件，开头几个字节（图像是16，标签是8）存储了一些我们不需要的信息（如魔数、图像数量等）**，如果不跳过**后续处理会出错**
#### 函数编写
知道了以上三个函数后，以下读取函数是易懂的：
```Python
def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(-1, 28, 28)
```

**注意：训练的时候只需要调用****训练集****（也就是train开头的两个文件），不要调用测试集！**

其中：

第二行等同于`f = gzip.open(file_path, 'rb')` ，但是with as语句确保了文件打开后关闭

第三、四行等同于`images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)`，这里先读取丢弃前16位字符起到了一样的作用

第五行，假如原一维 images 包含了N张图像数据，则长度为N * 28 * 28，那么将其reshape为一个(N, 28, 28)的三维数组；第一维参数 -1 表示该维度长度由 Numpy 根据原数组的元素数量和其他指定维度的大小自行计算

对于标签文件的读取同理，唯二的区别在于**只丢弃前8位**和**返回时不需要****reshape**

### 可视化

为了可视化，首先需要下载matplotlib库，这显然是大家都会的，忘记的话回顾 Lab 1.1

（准确而言，需要使用的是`matplotlib.pyplot`，以下简称为plt）

需要用到以下函数：
#### plt.imshow

```Plaintext
plt.imshow(image, cmap='gray')
```
#### plt.title()

```Plaintext
plt.title('Your Title Here')
```

#### plt.axis()

```Plaintext
plt.axis('off')
```

#### plt.show()

```Plaintext
plt.show()
```

#### 函数编写

那么以下函数是显然的

```Python
def display_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()
```

第三行使用格式化字符串来动态插入内容，其他易懂

需要说明：这里的参数 image 和 label 是单张图片的

例如传入images[0]，和label[0]，你会看到：

![](https://diangroup.feishu.cn/space/api/box/stream/download/asynccode/?code=MTcxYmVkNGNjMTVhZjdkZTkxOGU5ZjdjMTZhMDk1Y2ZfSE8zeDh5RlF2bU91RXhBS3BPa2NPRnBCOTVmbTd3cmtfVG9rZW46TFZzYmJZWkRXb2tZdWd4VUp4WmMxTllnbkljXzE3MjI3ODAyMDc6MTcyMjc4MzgwN19WNA)

### 数据预处理

#### 二值化

即：判断每个像素灰度是否大于某个阈值，大的一律纯黑（255），小的纯白（0）

```Python
def binarize(image, threshold=128):
    return (image > threshold).astype(np.uint8) * 255
```

- **逻辑**:
    
    - `image > threshold`:**通过广播机制**，将阈值扩大为一个数组和image比较，从而创建一个布尔数组，表示图像中每个像素是否大于阈值。
        
    - `.astype(np.uint8)`: 将布尔数组转换为无符号 8 位整数（0 或 1）。（astype是numpy内置函数，有用的地方在于它是**将一个数组进行数据转换**，不再只能转换单个数据）
        
    - `* 255`: 将 0 和 1 转换为 0 和 255，生成二值化图像。
        

#### 规范化和标准化

按学习文档，规范化和标准化指

_将图像__像素__值从 [0, 255] 缩放到 [0, 1] 或 [-1, 1]，或者进行 Z-score 标准化（减去平均值，除以__标准差__）_

规范化对大家是简单的：

```Python
def normalize(image):
    return image / 255.0
```

Z-core标准化对大家也是简单的

但我希望通过介绍标准化来介绍两个numpy函数：mean和std

```Python
def z_score_normalize(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    return (image - mean) / std_dev
```

  

#### 缩放

首先需要下载`scimage`库

**（不要像愚蠢的我一样****`conda install scimage`****然后浪费半个小时，请****`conda install scikit-image`****）**

（用`OpenCV`或许也行，但这里只介绍`scimage`的用法）

##### 插值法简介

缩放的基本原理是，根据缩放比例，根据原始图像一个或几个像素，计算出目标图像对应像素的值（即插值）

据gpt整理，常见插值法包括：

###### 1 最近邻插值

- **原理**: 对于每个新像素，选择与之最接近的原始像素值。
    
- **优点**: 简单快速，计算量小。
    
- **缺点**: 可能产生锯齿状边缘，图像质量较差。
    

###### 2 线性插值

- **原理**: 根据新像素在原始像素之间的位置，使用线性公式计算新像素值。
    
- **优点**: 生成的图像较为平滑。
    
- **缺点**: 对于大幅缩放，可能导致模糊。
    

###### 3 双线性插值

- **原理**: 在水平和垂直方向上进行线性插值，结合四个相邻像素的值。
    
- **优点**: 提供更平滑的结果，适合大多数应用。
    
- **缺点**: 计算复杂度高于最近邻和线性插值。
    

###### 4 立方插值

- **原理**: 使用相邻的16个像素（4x4区域）进行插值，生成更高质量的图像。
    
- **优点**: 能更好地保留细节和边缘。
    
- **缺点**: 计算量大，处理速度较慢。
    

##### skimage.transform.resize

_（原型有点复杂，看看就行）_

在以下参数中，**需要用到的是image、output_shape、order、anti_aliasing**

```Python
skimage.transform.resize(image, output_shape, order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=False)
```

1. **image**:
    
    1. 输入图像，通常是一个 NumPy 数组。
        
    2. 支持多通道（如 RGB）和灰度图像。
        
2. **output_shape**:
    
    1. 目标输出尺寸，格式为 `(height, width)` 或 `(depth, height, width)`（对于 3D 图像）。
        
3. **order**:
    
    1. 插值顺序，默认为 1（线性插值）。
        
    2. 可以选择 0（最近邻）、1（双线性）、2（双二次）等。
        
4. **mode**:
    
    1. 边界处理模式，定义图像边界的行为。
        
    2. 选项包括 `'reflect'`、`'constant'`、`'edge'`、`'symmetric'` 等。
        
5. **cval**:
    
    1. 当 `mode` 为 `'constant'` 时，指定常量值。
        
6. **clip**:
    
    1. 如果为 `True`，输出值会被限制在图像数据类型的范围内。
        
7. **preserve_range**:
    
    1. 如果为 `True`，在缩放时保持输入图像的数值范围。
        
8. **anti_aliasing**:
    
    1. 如果为 `True`，在缩放时启用抗锯齿处理，减少高频噪声
        

##### 函数编写

```Python
def resize_image(image, new_size=(28, 28)):
    return resize(image, new_size, anti_aliasing=True)
```

函数的new_size等于没缩放，自行调整

函数采用了抗锯齿，可以关闭

函数默认使用双线性插值，可以通过order修改

#### 裁剪

数组切片，大家都学过，不再赘述

```Python
def crop_image(image, top, left, height, width):
    return image[top:top + height, left:left + width]
```

  

### 完整代码

注：

缩放部分扩大四倍

裁剪部分裁剪左上角

**路径是我自己的路径**

```Python
import numpy as np
import gzip
import matplotlib.pyplot as plt
from skimage.transform import resize


def load_mnist_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(-1, 28, 28)


def load_mnist_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def display_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()


def binarize(image, threshold=128):
    return (image > threshold).astype(np.uint8) * 255


def z_score_normalize(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    return (image - mean) / std_dev


def resize_image(image, new_size=(28*2, 28*2)):
    return resize(image, new_size, anti_aliasing=True)


def crop_image(image, top, left, height, width):
    return image[top:top + height, left:left + width]


if __name__ == "__main__":
    images = load_mnist_images(r'C:\Users\24800\Desktop\AI\mnist\train-images-idx3-ubyte.gz')
    labels = load_mnist_labels(r'C:\Users\24800\Desktop\AI\mnist\train-labels-idx1-ubyte.gz')

    display_image(images[0], labels[0])
    display_image(binarize(images[0]), labels[0])
    display_image(z_score_normalize(images[0]), labels[0])
    display_image(resize_image(images[0]), labels[0])
    display_image(crop_image(images[0], 0, 0, 14, 14), labels[0])
```
# KNN算法
## 算法步骤
简单来说，KNN算法分为三个步骤：
- 步骤一：计算距离计算未知数据点与所有已知数据点之间的距离。
    
- 步骤二：找到最近的 K 个邻居根据计算出的距离，确定距离最近的 K 个邻居。
    
- 步骤三：进行投票或平均在分类任务中，根据最近的 K 个邻居的已知类别进行投票，类别出现频率最高的为预测结果；在回归中，则通常是这些邻居值的平均。
## 完整代码
```python
from collections import Counter#用于统计出现次数最多的标签

class KNN:
    def __init__(self, k):

        self.k = k
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):#整合动作
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    def _predict(self, x):#基础动作
        # 计算距离，存在diantances列表中
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # 获取k个最近的标签
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 多数表决
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
```
- `fit`方法用于训练模型，实际上只是将训练数据`X_train`和对应的标签`y_train`存储在类的实例变量中。

### 使用到的第三方库：
#调包侠
#### numpy
- `np.linalg.norm`用于计算向量的范数（即长度）。默认情况下，`np.linalg.norm`计算的是欧几里得范数（L2范数），即两个向量之间的欧几里得距离。
- `np.argsort`对距离进行排序，并取出前`k`个最小距离对应的索引,==注意是索引==
#### collections
- `Counter(k_nearest_labels).most_common(1)`用于统计出现次数最多的标签,`Counter`是`collections`模块中的一个类，用于统计可迭代对象中每个元素的出现次数。生成一个字典，键是标签，值是该标签在列表中出现的次数。
- `most_common`是`Counter`对象的一个方法，用于返回出现次数最多的前n个元素及其出现次数。参数`1`表示返回出现次数最多的一个元素及其出现次数。返回结果是一个列表，列表中的每个元素是一个元组，元组的第一个元素是标签，第二个元素是该标签出现的次数。
# 测试与评估
## 结果评估
```python
#预测结果评估函数
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
```
### 使用到的第三方库：
#调包侠 
#### sklearn.metrics
- `sklearn.metrics`是scikit-learn库中的一个模块，提供了多种用于评估机器学习模型性能的度量函数。这些度量函数可以用于分类、回归、聚类等不同类型的机器学习任务。
## 分类结果
![[Pasted image 20240902200618.png]]