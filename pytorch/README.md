https://colab.research.google.com/drive/1Z6K6nwbb69XfuInMx7igAp-NNVj_2xc3?usp=sharing#scrollTo=lsjIW9I_ztiO



# tensor和array的区别

Tensor和Array（通常指的是多维数组，比如NumPy中的ndarray）是数据结构，它们用于在计算机中存储数据。虽然这两个词在某些上下文中可以互换使用，但在深度学习框架（如PyTorch和TensorFlow）中，它们具有一些特定的差异：

1. **Tensor**:
   - 在深度学习框架中，Tensor通常是一个多维数组，它不仅可以存储数据，还可以在GPU或其他专门的硬件上进行运算。
   - Tensor通常有自动微分的能力，意味着它们可以自动跟踪它们在计算图中的操作，从而在进行反向传播时自动计算梯度。
   - Tensors被优化用于大规模**并行处理**，这对于深度学习中的大量矩阵运算非常重要。
2. **Array**:
   - Array通常指在普通编程语言（如Python中的NumPy库）中的多维数组数据结构。
   - 它们可能不支持GPU加速和自动微分等功能，或者这些功能不像深度学习框架中的Tensor那么易于使用。
   - 数组主要用于数据处理、数值计算，以及通用的多维数据存储。

总之，Tensor和Array在本质上都是多维数据的集合。但是，"Tensor"这个术语在深度学习上下文中通常与硬件加速和自动微分功能联系在一起，而“Array”则更多用于指代普通编程环境中的多维数据结构。



# torch基础

## 创建torch

### from a numpy array

```python
# Initialize a tensor from a Python List
data = [
        [0, 1], 
        [2, 3],
        [4, 5]
       ]
x_python = torch.tensor(data)

# Print the tensor
x_python

>>>
tensor([[0, 1],
        [2, 3],
        [4, 5]])
```

```python
npArray = np.array(data)
x_tensor = torch.from_numpy(npArray)
```

### from a tensor

We can also initialize a tensor from another tensor, using the following methods:

- `torch.ones_like(old_tensor)`: Initializes a tensor of `1s`.
- `torch.zeros_like(old_tensor)`: Initializes a tensor of `0s`.
- `torch.rand_like(old_tensor)`: Initializes a tensor where all the elements are sampled from a uniform distribution between `0` and `1`.
- `torch.randn_like(old_tensor)`: Initializes a tensor where all the elements are sampled from a normal distribution.

```python
x = torch.tensor([[1.,2],[3,4]])
>>tensor([[1., 2.],
        [3., 4.]])
```

```python
x_zeros = torch.zeros_like(x)
>>
x_zeros
tensor([[0., 0.],
        [0., 0.]])
```

```python
x_ones = torch_ones_like(x)
>>
tensor([[1., 1.],
        [1., 1.]])
```

```pyhton
# between 0 and 1
x_rand = torch.rand_like(x)
x_rand
tensor([[0.8979, 0.7173],
        [0.3067, 0.1246]])
```

```python
##服从标准正态分布
# Initialize a tensor where each element is sampled from a normal distribution
x_randn = torch.randn_like(x)
x_randn
tensor([[-0.6749, -0.8590],
        [ 0.6666,  1.1185]])
```



## tensor 矩阵之间运算

确实，`@` 运算符、`*` 运算符和 `dot` 函数在 Python 和其科学计算库中有特定的用途。

1. **`@` 运算符**:
   - 在 Python 中，`@` 是矩阵乘法运算符，它用于执行两个二维数组（或更高维）之间的矩阵乘法。这意味着它会进行行与列的点积运算，与线性代数中的矩阵乘法相同。
   - 例如，`numpy`、`scipy` 和深度学习库如 `PyTorch` 和 `TensorFlow` 都支持使用 `@` 进行矩阵乘法。
   - 如果矩阵 A 的维度是 (m, n) 而矩阵 B 的维度是 (n, p)，则 A @ B 的结果将是一个 (m, p) 维的矩阵。

2. **`*` 运算符**:
   - 在 Python 的 `numpy` 和 `array` 类型中，`*` 表示元素间乘法（也称作**哈达玛**积或元素积），它将两个数组的对应元素相乘，并返回一个数组。
   - **这意味着两个数组必须有相同的大小或者满足广播规则。**

3. **外积**:
   - 外积（也称为外积或向量积）通常指两个向量的笛卡尔积，其结果是一个矩阵。
   - 在 `numpy` 中，你可以使用 `numpy.outer()` 函数来计算向量的外积。

4. **`numpy.dot()` 函数**:
   - 在 `numpy` 中，`dot()` 既可以执行两个向量的点积，也可以执行两个矩阵的矩阵乘法。当对两个向量使用时，它计算的是点积，结果是一个标量。当对两个矩阵使用时，它执行的是矩阵乘法，结果是一个矩阵。
   - **对于二维数组（即矩阵），`numpy.dot()` 和 `@` 运算符的效果是相同的。**

简而言之，`@` 运算符用于矩阵乘法，`*` 运算符用于数组的元素积，外积是一个特殊的向量运算，而 `numpy.dot()` 根据输入的维度（一维或二维）可以计算点积或矩阵乘法。

### @ 矩阵乘法

```python
# Create a 4x3 tensor of 6s
a = torch.ones((4,3)) * 6
>>>a
tensor([[6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.]])

# Create a 1D tensor of 2s
b = torch.ones(3) * 2
>>>b
tensor([2., 2., 2.])


a / b
>>>
tensor([[3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.]])


# Alternative to a.matmul(b)
# a @ b.T returns the same result since b is 1D tensor and the 2nd dimension
# is inferred
a  @ b
>>>
tensor([36., 36., 36., 36.])
```





## tensor‘s shape

打印shape是调试的好方法

```python
pp.pprint(a.shape)
pp.pprint(a.T.shape)
>>>
torch.Size([4, 3])
torch.Size([3, 4])
```





### 重塑tensor的方法 view

```python
x = torch.arange(1,7)
x_view = x.view(2,3)
x_view = x.view(2,-1)  //自动计算
>>>
tensor([[1, 2, 3],
        [4, 5, 6]])
```



## tensor内部运算

### sum

```python
x = torch.arange(1,7)
x_view = x.view(2,3)
pprint.pprint(x_view)
pprint.pprint(x_view.sum(dim=0))  //按列求和
pprint.pprint(x_view.sum(dim=1))  //按行求和
>>>
tensor([[1, 2, 3],
        [4, 5, 6]])
tensor([5, 7, 9])
tensor([ 6, 15])

```

### std 标准差

```python
pprint.pprint(x_view.std(dim=0))  
pprint.pprint(x_view.std(dim=1))  

>>>
tensor([2.1213, 2.1213, 2.1213])
tensor([1., 1.])
```



### mean 均值

```python
# Create an example tensor
m = torch.tensor(
    [
     [1., 1.],
     [2., 2.],
     [3., 3.],
     [4., 4.]
    ]
)

pp.pprint("Mean: {}".format(m.mean()))
pp.pprint("Mean in the 0th dimension: {}".format(m.mean(0)))
pp.pprint("Mean in the 1st dimension: {}".format(m.mean(1)))
>>>
'Mean: 2.5'
'Mean in the 0th dimension: tensor([2.5000, 2.5000])'
'Mean in the 1st dimension: tensor([1., 2., 3., 4.])'
```



## tensor  矩阵的拼接 cat

```python
x = torch.arange(1,7)
a = x.view(2,3)*1.0
a_cat0 = torch.cat([a,a,a],0)
a_cat1 = torch.cat([a,a,a],1)
pprint.pprint(a)
pprint.pprint(a_cat0)
pprint.pprint(a_cat1)
>>>
tensor([[1., 2., 3.],
        [4., 5., 6.]])
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [1., 2., 3.],
        [4., 5., 6.],
        [1., 2., 3.],
        [4., 5., 6.]])
tensor([[1., 2., 3., 1., 2., 3., 1., 2., 3.],
        [4., 5., 6., 4., 5., 6., 4., 5., 6.]])
```



## 索引indexing

```python
# Initialize an example tensor
x = torch.Tensor([
                  [[1, 2], [3, 4]],
                  [[5, 6], [7, 8]], 
                  [[9, 10], [11, 12]] 
                 ])
>>>x
tensor([[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]],

        [[ 9., 10.],
         [11., 12.]]])
```

```python
x.shape
>>>
torch.Size([3, 2, 2])
```

```python
x[0] ##第0个维度的子张量
>>>
tensor([[1., 2.],
        [3., 4.]])
```

```python
x[:, 0, 0]  ##所有维度的子张量的第0行第0列
>>>
tensor([1., 5., 9.])
```

```python
i = torch.tensor([0, 0, 1, 1])
x[i]##i作为索引 belike:  x[0] x[0] x[1] x[1]
>>>
tensor([[[1., 2.],
         [3., 4.]],

        [[1., 2.],
         [3., 4.]],

        [[5., 6.],
         [7., 8.]],

        [[5., 6.],
         [7., 8.]]])

```

![image-20240312181128980](README.assets/image-20240312181128980.png)

## 获得标量 item

```python
x[0, 0, 0]
>>>tensor(1.)
```

```python
x[0, 0, 0].item()
>>>1.0
```





# autograd 自动梯度计算

PyTorch 和其他机器学习库以其自动微分功能而闻名。也就是说，只要我们定义了需要执行的操作集合，框架本身就可以计算出如何计算梯度。我们可以调用 backward() 方法来请求 PyTorch 计算梯度，这些梯度随后存储在 grad 属性中。

```python
# Create an example tensor
# requires_grad parameter tells PyTorch to store gradients
x = torch.tensor([2.], requires_grad=True)##默认情况下关闭，开启之后会进行记录，消耗算力，因为我们只有在训练更新的时候才需要梯度，当我们真正去预测的时候是不需要梯度去额外小号算力的

# Print the gradient if it is calculated
# Currently None since x is a scalar  因为只有一个元素被认为是标量
pp.pprint(x.grad)
>>>none
```

```python
# Calculating the gradient of y with respect to x
y = x * x * 3 # 3x^2
y.backward() #simply，y是cost function
pp.pprint(x.grad) # d(y)/d(x) = d(3x^2)/d(x) = 6x = 12
>>>tensor([12.])
```

```python
z = x * x * 3 # 3x^2
z.backward()
pp.pprint(x.grad)
>>>tensor([48.])  
```

我们可以看到，**x.grad 被更新为到目前为止计算的梯度之和**。当我们在神经网络中运行反向传播时，**我们会在进行更新之前对某个神经元的所有梯度求和**。这正是此处发生的情况！这也是为什么我们需要在每次训练迭代中运行 zero_grad() 的原因（稍后会详细介绍）。否则，我们的梯度会从一个训练迭代累积到另一个训练迭代，这将导致我们的更新出错。

**如果 `y` 不是标量而是一个向量或矩阵**，我们需要传递一个与 `y` 形状相同的张量作为 `backward()` 方法的参数，以指定对 `y` 中每个元素的梯度权重。例如：

```python
# 创建一个二维张量
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

# 定义一个操作
y = x ** 2  # y = x^2

# 创建一个与 y 形状相同的张量，表示对 y 中每个元素的梯度权重
grad_tensor = torch.tensor([[1., 1.], [1., 1.]])

# 对 y 进行反向传播，计算 x 的梯度
y.backward(grad_tensor)

# 打印 x 的梯度
print(x.grad)  # 输出 [[2., 4.], [6., 8.]]
```

**上游梯度、本地梯度、下游梯度是针对某个神经元的**

**但是我们所谓的梯度这个概念，其实是针对所有神经网络的参数**



# Nerual Network Module

但是接下来，我们将使用 PyTorch 中的 torch.nn 模块中的预定义模块。然后我们将把这些模块组合起来创建复杂的网络。

```
import torch.nn as nn 
```



## Liner layer

我们首先要理解几个概念

### 样本 特征向量 特征

在深度学习中，数据通常以张量的形式组织和处理。在这个例子中，输入张量 `input` 的形状为 `(2, 3, 4)`，可以从以下三个角度理解：

1. **样本（Sample）**：
   - 样本是指单个数据点或观察值。在这个例子中，张量的第一个维度大小为 2，意味着有 2 个样本。在深度学习中，一个样本通常对应于一个训练实例，如一张图片、一个文本句子或一个数据记录。
2. **特征向量（Feature Vector）**：
   - 特征向量是指描述样本的一组数值。在这个例子中，每个样本有 3 个特征向量，即张量的第二个维度大小为 3。这可以理解为每个样本由 3 个不同的特征向量组成，每个特征向量代表样本的不同方面或特性。
3. **特征（Feature）**：
   - 特征是特征向量中的单个数值。在这个例子中，每个特征向量有 4 个特征，即张量的第三个维度大小为 4。这意味着每个特征向量由 4 个数值组成，每个数值代表一个特定的特征或属性。

总的来说，这个张量可以被看作是包含 2 个样本的批次，每个样本有 3 个特征向量，每个特征向量有 4 个特征。这种组织数据的方式在深度学习中很常见，因为它允许模型批量处理多个样本，从而提高计算效率。





我们可以使用 **nn.Linear(H_in, H_out)** 来创建一个线性层。**这将接收一个维度为 (N, *, H_in) 的矩阵，并输出一个维度为 (N, *, H_out) 的矩阵**。星号 (*) 表示中间可以有任意数量的维度。线性层执行 Ax+b 操作，其中 A 和 b 被随机初始化。如果我们不希望线性层学习偏置参数，我们可以使用 bias=False 来初始化我们的层。

```python
input = torch.ones(2,3,4)
pp.print(input)
>>>
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])


# Make a linear layers transforming N,*,H_in dimensinal inputs to N,*,H_out
# dimensional outputs
linear = nn.Linear(4, 2)##创建/实例化线性层##输入N_in,N_out 对应特征值4，2输入特征值为4的张量，输出特征值为2的张量
linear_output = linear(input)  
linear_output
>>>
tensor([[[ 0.1582, -0.4119],
         [ 0.1582, -0.4119],
         [ 0.1582, -0.4119]],

        [[ 0.1582, -0.4119],
         [ 0.1582, -0.4119],
         [ 0.1582, -0.4119]]], grad_fn=<ViewBackward0>)
##从shape 2,3,4->2,3,2
```

### grad_fn

***在 PyTorch 中，`grad_fn` 属性表示创建给定张量的函数，它是计算图中的一个节点。`grad_fn` 跟踪了张量是如何被计算出来的，这对于自动梯度计算（自动微分）是必要的。在反向传播过程中，PyTorch 会使用这些 `grad_fn` 来计算梯度。***

***例如，如果一个张量是通过一个操作生成的，如 `view` 操作（用于改变张量的形状），那么它的 `grad_fn` 将被标记为 `<ViewBackward>`，表示这个张量是通过 `view` 操作得到的。`<ViewBackward0>` 只是表示这是 `ViewBackward` 类的一个实例***

***在 PyTorch 中，知道一个张量是通过 `view` 操作（或类似的操作，如 `reshape`、`transpose` 等）得到的对计算梯度是有帮助的，因为这些操作会改变张量的形状而不改变其数据内容。当你对一个经过这些操作的张量进行反向传播时，PyTorch 需要确保梯度正确地传递回原始张量，即使它们的形状不同。***



```python
list(linear.parameters()) # Ax + b ##其实是A.T
>>> 输出两个张量 第一个是权重矩阵 第二个是偏置矩阵
[Parameter containing:
 tensor([[-0.0675,  0.0311, -0.0579,  0.4879],
         [-0.1701, -0.4245, -0.3129,  0.0659]], requires_grad=True),
 Parameter containing:
 tensor([-0.2354,  0.4296], requires_grad=True)]
```

我们通过输入特征值为4转化成特征值为2，过程为：

对于第一个样本，特征向量[1,1,1,1]xA.T+B=[y[0],y[1]]  

特征向量1x4，A.T形状4x2->生成1x2的矩阵特征值为2，实现liner层

```PYTHON
y[0] = (-0.0675 * 1) + (0.0311 * 1) + (-0.0579 * 1) + (0.4879 * 1) + (-0.2354)
      = -0.0675 + 0.0311 - 0.0579 + 0.4879 - 0.2354
      = 0.1582

y[1] = (-0.1701 * 1) + (-0.4245 * 1) + (-0.3129 * 1) + (0.0659 * 1) + (0.4296)
      = -0.1701 - 0.4245 - 0.3129 + 0.0659 + 0.4296
      = -0.4119
```

