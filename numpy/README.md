pycharm快速向下复制一行 ctrl+d

删除一行 ctrl+y

# 数组基础

## 1.1数据类型

### 1.1.1整数型数组与浮点型数组

numpy一种数组只能存放一种数据类型，用于节省内存

```python
import numpy as np
arr1 = np.array([1,2,3])
print (arr1)
>>> [1 2 3]  //numpy数组没有逗号
```

```python
import numpy as np
arr1 = np.array([1.1,2,3])
print (arr1)
>>>[1.1 2.  3. ] 
```



### 1.1.2 同化定理

往整数型数组添加浮点数，浮点数会截断

往浮点型数组添加整数，整数会升级为浮点

```python
import numpy as np
arr1 = np.array([1.1,2,3])
arr1[0]=100
print (arr1)
>>>[100.   2.   3.]
```

```python
import numpy as np
arr1 = np.array([1,2,3])
arr1[0]=100.1
print (arr1)
>>>[100   2   3]
```

​			

### 1.1.3 共同改变定理

整数型数组和浮点型数组相互转换 使用.astype()方法,返回转化后的数组，原数组不变
**np.后面是函数 对象后面是方法**

```python
import numpy as np
arr1 = np.array([1,2,3])
arr2 = arr1.astype(float)
print (arr2)
```

```python
import numpy as np
arr1 = np.array([1.1,2.2,3])
arr2 = arr1.astype(int)
print (arr2)
>>>[1 2 3]
```

除了上面的方法，还有不经意间升级

```python
//整数型数组和浮点数运算
import numpy as np
arr1 = np.array([1,2,3])
print (arr1*1.0)   //numpy中数组每个元素都乘以1
>>>[1. 2. 3.]
//整数型数组除法，即使除以整数也会变成浮点数数组
import numpy as np
arr1 = np.array([1,2,3])
print (arr1/1)
>>>[1. 2. 3.]
//整数型数组和浮点数数组运算
import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([1.1,2.1,3.1])
print (arr1+arr2)
>>>print (arr1+arr2)
```





## 1.2数组维度

### 1.2.1 一维数组和二维数组

- 不同维度数组的本质区别

  几维数组几个中括号包裹

- 有些函数需要传入数组的形状参数，不用维度数组的形状参数为

  一维数组的形状参数：(x,)或x

  二维数组：(x,y)

  三维数组：(x,y,z)

- 同一序列举例

  当数组有一层中括号，如[1,2,3],形状参数3或者(3,) 代表三行

  [[1,2,3]]，二维数组，形状参数(1,3)一行三列

  [[[1,2,3]]]，三维数组(1,1,3)

这里用np.ones来讲解，因为这个函数要传入形状参数

```python
import numpy as np
arr1 = np.ones(3)
print(arr1)
>>>[1. 1. 1.]

import numpy as np
arr1 = np.ones((1,3))
print(arr1)
>>>[[1. 1. 1.]]

import numpy as np
arr1 = np.ones((1,1,3))
print(arr1)
>>>[[[1. 1. 1.]]]
```

也可以用.shape属性来查看形状

```python
import numpy as np
arr1 = np.ones((1,1,3))
print(arr1.shape)
>>>(1, 1, 3)
```

### 1.2.2不同维度数组之间的转换

数组方法.reshape()，需要传入重塑后的形状参数

```python
//一维转二维
import numpy as np
arr1 = np.arange(10)
arr2 = arr1.reshape((1,-1))
print(arr2)
>>>一维数组转成二位数组，固定行数之后，列数为-1，可以自动计算出列数，即1*10（原一维数组个数）/1（转换后行数）=列数

//多维转一维
import numpy as np
arr1 = np.arange(10).reshape((2,-1))
arr2 = arr1.reshape(-1)  //给个-1 就是转一维
print(arr2)
>>>[0 1 2 3 4 5 6 7 8 9]
```



**今后一维数组称为向量，二维数组称为矩阵**





# 数组的创建

## 2.1创建指定数组

明确知道每一个元素的具体值时，使用np.array()函数进行创建，将python列表转化成np数组

```python
arr1 = np.array([1,2,3])//创建向量
arr2 = np.array([[1,2,3]]) //创建行矩阵
arr3 = np.array([ [1],[2],[3] ]) //创建列矩阵  一共中括号里中有几个元素就是激烈，有几个中括号就是几行
arr4 = np.array([ [1,2,3],[4,5,6] ])


```

## 2.2创建递增数组

np.arange()

```python
arr1 = np.arange(10)
print(arr1)
>>>[0 1 2 3 4 5 6 7 8 9]

arr2 = np.arrange(10,20)
print(arr2)
>>>[10 11 12 13 14 15 16 17 18 19]

import numpy as np
arr1 = np.arange(10,20,2)  //步长是2
print(arr1)
>>>[10 12 14 16 18]  


浮点数
arr1 = np.arange(10,20,2)*1.0  or  任意参数变成浮点数


np.arrange(1,17).reshape(4,4)
```



## 2.3 创建同值数组

```python
//全0数组
arr1 = np.zeros(4)  形状为4的向量
print(arr1)
>>>[0. 0. 0. 0.]

arr1.ones((1,2))  //形状为（1，2）的全1矩阵
print(arr1)  
>>>[[1. 1.]]


arr1 = 3.14*np.ones((2,3))
print(arr1)
>>>[[3.14 3.14 3.14]
 [3.14 3.14 3.14]]
```



## 2.4创建随机数组

### 2.4.1浮点型随机数组

```python
arr1 =np.random.random(5)  //随机数范围[0,1)
print(arr1)
>>>[0.26688669 0.08056473 0.83595846 0.94521307 0.67801163]

创建一个60-100范围内均匀分布的3行3列随机数组
arr1 =(100-60)*np.random.random((3,3))+60 //先生成0-40再加60
print(arr1)
```



### 2.4.2 整数型随机数组

```python
arr1 =np.random.randint(10,100,(1,15))
print(arr1)
>>>[[64 90 19 94 87 59 70 84 37 23 24 67 45 48 40]]
```



### 2.4.3服从正态分布的随机数组、

```python
arr1 =np.random.normal(4,2,(2,5)) //均值，标准差，形状
print(arr1)
>>>[[2.41062    3.05075851 5.06797223 5.13336656 1.68124686]
 [2.72032718 5.42585309 1.35236911 3.27317787 1.88057483]]

//如果是标准正态分布(0,1)
arr1 =np.random.randn((2,5))
[[-0.2357357  -1.84865885  1.52610709 -0.79777238  0.44978987]
 [-2.12101013  0.24850036 -0.02268558 -1.02203212  0.91009054]]
```





# 数组的索引

## 3.1访问数组元素

### 3.1.1访问向量

```python
arr1 = np.arange(1,10) [1,9]
print(arr1[3]) //正着访问 >>>3
print(arr1[-1]) //到这访问 >>>9

//修改元素
arr1[3] = 999
print(arr1)
>>>[  1   2   3 999   5   6   7   8   9]
```



### 3.1.2访问矩阵

```python
arr1 = np.array([ [1,2,3],[4,5,6] ])
print(arr1)
print(arr1[0,2])  >>>3
print(arr1[1,-2]) >>>5

//修改元素
arr1[1,0] = 100.9
import numpy as np
arr1 = np.array([ [1,2,3],[4,5,6] ])
arr2 = arr1.astype(float)
arr2[1,0] = 100.9
print(arr2)
>>>[[  1.    2.    3. ]
 [100.9   5.    6. ]]
```



## 3.2花式索引

### 3.2.1 向量的花式索引

```python
arr1 = np.arange(0,90,10)
print(arr1[ [0,2] ])  //第一列第三列
>>>[ 0 20]
```

### 3.2.2矩阵的花式索引

取出来还是矩阵

```python
arr1 = np.arange(1,17).reshape(4,4)
print(arr1)
print(arr1[ [0,2],[0,1] ])  [行],[列]
>>[ 1 10]   1->(第0行第0列) 10->第2行第1列

//可以做到修改多个数组元素
arr1 = np.arange(1,17).reshape(4,4)
arr1[ [0,2],[0,1] ]=100
print(arr1)
>>>[[100   2   3   4]
 [  5   6   7   8]
 [  9 100  11  12]
 [ 13  14  15  16]]
```



## 3.3访问数组切片

### 3.3.1向量的切片

左开右闭

```python
arr1 = np.arange(10)
print(arr1[1:4]) >>[1 2 3] 
print(arr1[1:])  >>[1 2 3 4 5 6 7 8 9]
print(arr1[:4])  >>[0 1 2 3]

print(arr1[2:-2]) //[2 3 4 5 6 7]
print(arr1[:-2]) //[0 1 2 3 4 5 6 7]

```

```python
print(arr1)
print(arr1[::2])  //开头到结尾 每两个元素采样一次（隔一个）
print(arr1[::3])
print(arr1[1:-1:2])
>>>[0 1 2 3 4 5 6 7 8 9]
>>>[0 2 4 6 8]
>>>[0 3 6 9]
>>>[1 3 5 7]
```



### 3.3.2矩阵的切片

```
arr1 = np.arange(1,21).reshape(4,5)
print(arr1)
print(arr1[1:3,1:-1]) //行数1-2 列数1-倒数1
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]
 [16 17 18 19 20]]
[[ 7  8  9]
 [12 13 14]]
print(arr1[::3, ::2]) 每隔两行采样一次 每隔一列采样一次
>>>[[ 1  3  5]
 [16 18 20]]

```

### 3.3.3提取矩阵的行

```python
arr1 = np.arange(1,21).reshape(4,5)
print(arr1)
print(arr1[[2]])  //提取的还是矩阵 
print(arr1[2,:])  //提取出一行的向量 第2行
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]
 [16 17 18 19 20]]
[[11 12 13 14 15]]
[11 12 13 14 15]
```

提取行可以简写不写：

### 3.3.4提取矩阵的列

```python
arr1 = np.arange(1,21).reshape(4,5)
print(arr1)
print(arr1[:,1:3])  //提取1-2列
>>>[[ 2  3]
 [ 7  8]
 [12 13]
 [17 18]]
print(arr1[:,1]) //[ 2  7 12 17]这里是个向量了 目的是为了节省空间 
为了把这个向量转化为矩阵
可以
cut = arr1[:,1].reshape(1,-1) >>>[[2 7 12 17]]
print(cut.T) //转置 
[[ 2]
 [ 7]
 [12]
 [17]]

或者直接cut = arr1[:,1].reshape(-1，3)
```



## 3.4数组切片仅是视图

### 3.4.1 数组切片仅是视图

就是浅拷贝的意思

```python
arr1 = np.arange(10)
cut = arr1[:3]
cut[0] = 100
print(arr1)
>>>[100   1   2   3   4   5   6   7   8   9]
```

想为切片产生新的内存空间，就是用.copy()方法



## 3.5数组赋值仅是绑定

```
arr1 = np.arange(10)
arr2 = arr1
arr2[0] = 100
print(arr1)>>>[100   1   2   3   4   5   6   7   8   9]
```

也是使用.copy 创建新的变量