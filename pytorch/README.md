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

**在你的例子中，`x` 不是模型的参数，而是一个张量，它被用来计算另一个张量 `y` 的值。在这个特定的情况下，`x` 被用作一个自变量，用来演示如何计算 `y` 相对于 `x` 的梯度。正常情况下我们都是求模型参数的grad**

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

list(linear.parameters())   linear.parameters()返回一个生成器 list将他转换为列表

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





## activation function layer

我们还可以使用 nn 模块将激活函数应用于我们的张量。激活函数用于为我们的网络添加非线性。一些激活函数的示例包括 nn.ReLU()、nn.Sigmoid() 和 nn.LeakyReLU()。激活函数分别对每个元素进行操作，因此我们作为输出得到的张量的形状与我们传入的张量的形状相同



### Sigmoid(*x*)=1+*e*−*x*1

对于给定的线性输出 `liner_output`，Sigmoid 函数逐元素地应用这个公式，得到输出张量 `output`。

```pyhton
sigmnod = nn.Sigmoid()
output = sigmnod(liner_output)
>>>
tensor([[[0.7679, 0.4755],
         [0.7679, 0.4755],
         [0.7679, 0.4755]],

        [[0.7679, 0.4755],
         [0.7679, 0.4755],
         [0.7679, 0.4755]]], grad_fn=<SigmoidBackward0>)
```

### 

## putting the layers together

到目前为止，我们已经看到我们可以创建层并将一个层的输出作为下一个层的输入。我们可以使用 nn.Sequential 来代替创建中间张量并传递它们，它正是为此而设计的。



```python
block = nn.Sequential(
		nn.Linear(4,2),
		nn.Sigmoid()
)
input = torch.ones(2,3,4)
output = block(input)
>>>
tensor([[[0.6865, 0.3374],
         [0.6865, 0.3374],
         [0.6865, 0.3374]],

        [[0.6865, 0.3374],
         [0.6865, 0.3374],
         [0.6865, 0.3374]]], grad_fn=<SigmoidBackward0>)
```





## Custom Modules 自定义模块

### define模型

#### `nn.Module` 类

在 PyTorch 中，所有的神经网络模型都应该继承自 `nn.Module` 类。`nn.Module` 是 PyTorch 中所有神经网络模块的基类，它提供了一些基本的功能，如参数管理、模型训练和评估等。

当你创建一个自定义的神经网络模型时，你需要定义一个类来表示这个模型，并让这个类继承自 `nn.Module`。在这个类中，你需要实现两个主要的方法：`__init__` 和 `forward`。

#### `__init__` 方法

`__init__` 方法是类的构造函数，用于初始化模型的层和参数。在这个方法中，你通常会**调用超类 `nn.Module` 的构造函数**，这是通过 `super(MultilayerPerceptron, self).__init__()` 这行代码实现的。调用超类的构造函数是必要的，**因为它会初始化一些在 `nn.Module` 中定义的内部变量和结构，这对于后续的参数管理和模型训练是必要的。**

在 `__init__` 方法中，除了调用超类的构造函数外，**你还会定义模型中的各个层**。在你的例子中，这是通过创建一个 `nn.Sequential` 容器并添加线性层和激活函数来实现的。

总的来说，`__init__` 方法的作用是初始化模型的结构和参数，而调用超类 `nn.Module` 的构造函数是为了确保这些初始化工作符合 PyTorch 的要求

```python
class MultilayerPerceptron(nn.Module):

  def __init__(self, input_size, hidden_size):
    # Call to the __init__ function of the super class
    super(MultilayerPerceptron, self).__init__()

    # Bookkeeping: Saving the initialization parameters
    self.input_size = input_size 
    self.hidden_size = hidden_size 

    # Defining of our model
    # There isn't anything specific about the naming of `self.model`. It could
    # be something arbitrary.
    self.model = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.input_size),
        nn.Sigmoid()
    )
    
  def forward(self, x):
    output = self.model(x)
    return output
```

在 PyTorch 中，当你定义了一个模型继承自 `nn.Module` 并实现了 `forward` 方法后，你可以直接将模型实例当作一个函数来调用。当你这样做时，实际上是在调用模型的 `forward` 方法。也就是说，**`model(input)` 实际上是在后台调用 `model.forward(input)`。**

在你的例子中，`model = MultilayerPerceptron(5, 3)` 创建了一个 `MultilayerPerceptron` 的实例，其中输入层大小为 5，隐藏层大小为 3。当你执行 `model(input)` 时，它会将输入数据 `input` 通过模型的前向传播函数 `forward` 进行处理，并返回输出结果。这就是为什么你不需要直接调用 `forward` 方法；PyTorch 为你处理了这一部分。

####  `__call__` 方法

在 PyTorch 中，这种直接调用模型实例以进行前向传播的方式是通过 `nn.Module` 类的特殊方法 `__call__` 实现的。当你创建一个继承自 `nn.Module` 的类（如你的 `MultilayerPerceptron` 类）并实例化它时，你得到的对象是一个可调用对象。这意味着你可以像调用函数一样调用这个对象。

这是如何工作的：

1. **`__call__` 方法**：在 `nn.Module` 类中定义了一个 `__call__` 方法。这个方法使得任何继承自 `nn.Module` 的类的实例都可以像函数一样被调用。
2. **调用 `forward`**：当你调用模型实例时（例如，`model(input)`），实际上是在调用 `nn.Module` 中的 `__call__` 方法。这个 `__call__` 方法会做一些准备工作（如设置模型为训练模式或评估模式，应用钩子函数等），然后调用你在子类中定义的 `forward` 方法，并将输入参数传递给它。
3. **返回输出**：`forward` 方法处理输入并返回输出。这个输出就是 `model(input)` 的结果。

这种设计使得模型的使用更加直观和简洁。你不需要显式地调用 `forward` 方法；你只需要像调用普通函数一样调用模型实例即可。这也使得模型的使用与普通的 Python 函数调用保持一致，降低了学习门槛。多层感知机（Multilayer Perceptron, MLP）是一种前馈神经网络，它由一个输入层、一个或多个隐藏层和一个输出层组成。每一层都包含若干个神经元，相邻层之间的神经元通过权重连接。MLP 使用非线性激活函数来引入非线性特征，使得网络能够学习复杂的函数映射。

#### MLP 的主要作用包括：

1. **分类**：MLP 可以用于二分类或多分类问题。在输出层使用适当的激活函数（如 softmax）和损失函数（如交叉熵损失），MLP 可以学习将输入特征映射到不同的类别标签上。
2. **回归**：MLP 也可以用于回归问题，即预测连续值。通过在输出层使用线性激活函数（或不使用激活函数），MLP 可以学习输入特征与连续目标值之间的关系。
3. **特征提取**：隐藏层可以学习输入数据的高级特征表示，这些特征可以用于其他机器学习任务，如聚类或降维。
4. **近似任意函数**：理论上，具有至少一个隐藏层且足够多神经元的 MLP 能够近似任意连续函数，这使得它们在许多应用中非常灵活和强大。

```python
list(model.named_parameters())
>>> 两个线性层的参数
('model.0.weight',
  Parameter containing:
  tensor([[ 0.1085,  0.4337,  0.0553, -0.2615,  0.4020],
          [-0.0603,  0.3529,  0.4155, -0.0905,  0.0606],
          [ 0.4449,  0.0792, -0.1505, -0.4393,  0.2296]], requires_grad=True)),
 ('model.0.bias',
  Parameter containing:
  tensor([-0.4108, -0.2566,  0.1668], requires_grad=True)),
 ('model.2.weight',
  Parameter containing:
  tensor([[-0.0014,  0.5237,  0.2961],
          [-0.3127, -0.3627, -0.2712],
          [-0.4654, -0.3234,  0.0029],
          [ 0.1980, -0.3169,  0.3647],
          [-0.0985, -0.4923, -0.4869]], requires_grad=True)),
 ('model.2.bias',
  Parameter containing:
  tensor([ 0.3559,  0.2877,  0.0828,  0.0669, -0.5682], requires_grad=True))]
```





### optimization 优化

我们已经展示了如何使用 backward() 函数计算梯度。仅有梯度对于我们的模型来说是不够的，我们还需要知道如何更新模型的参数。这就是优化器发挥作用的地方。torch.optim 模块包含了几个我们可以使用的优化器。一些流行的例子包括 optim.SGD 和 optim.Adam。在初始化优化器时，我们传递我们的模型参数，可以通过 model.parameters() 访问，告诉优化器它将优化哪些值。**优化器还有一个学习率（lr）参数，决定了每一步将进行多大的更新。不同的优化器还有不同的超参数**。

```python
import torch.optim as optim
```



构建神经网络模型时，定义优化目标（即损失函数）的重要性。优化函数用于根据损失函数计算得到的梯度更新模型的参数，以减少损失并提高模型的性能。

1. **定义损失函数**：损失函数衡量模型预测与真实值之间的差异。你可以根据具体的任务自定义损失函数，或者使用 PyTorch 提供的预定义损失函数，例如 `nn.BCELoss()`（二元交叉熵损失），这在二分类问题中很常用。
2. **创建虚拟数据**：在实际应用模型之前，通常会使用一些虚拟数据来测试模型的结构和优化过程是否正确。这些虚拟数据应该与实际数据具有相同的格式和维度。
3. **综合运用**：一旦定义了优化函数和损失函数，并准备了数据，就可以将这些元素结合起来进行模型训练。这通常涉及到在训练循环中反复计算损失，使用优化器根据损失的梯度更新模型参数，并监控训练过程中的性能指标

#### 定义data

```python
# Create the y data
y = torch.ones(10, 5)

# Add some noise to our goal y to generate our x
# We want out model to predict our original data, albeit the noise
x = y + torch.randn_like(y)
x
>>>
tensor([[ 0.8380,  2.5777,  1.1166,  0.9116, -0.4451],
        [ 0.8303,  2.1061,  2.2382,  2.3162,  0.9996],
        [-0.5046,  1.1863,  1.2903,  1.5943,  1.0434],
        [-0.7874,  0.7638,  0.2501, -1.4281,  0.2135],
        [-0.5343,  0.9503,  2.0201,  0.5922,  0.8805],
        [ 2.8209,  1.2848, -0.6981,  1.0437, -0.1171],
        [ 2.2282,  0.9780,  2.1590, -0.5419,  2.4298],
        [ 1.6055, -0.1102,  0.5263,  1.8585,  1.9228],
        [-1.1403,  0.6889,  0.5987,  1.4502,  2.5326],
        [ 1.3639,  2.1660,  2.7522,  0.4086,  0.7330]])
```

#### 定义optimizer优化和损失函数

```python
# Instantiate the model
model = MultilayerPerceptron(5, 3)

# Define the optimizer
adam = optim.Adam(model.parameters(), lr=1e-1)

# Define loss using a predefined loss function
loss_function = nn.BCELoss()

# Calculate how our model is doing now
y_pred = model(x)
loss_function(y_pred, y).item()
>>>  一开始我疑问为什么lossfunction的结果是个张量标量，后来我意识到，之前说的应该是偏导和输入的矩阵格式相同，lossfunction代表的就是预测值和真实值的差异，所以应该就是一个数字
0.7544203996658325
```

#### train

```python
# Set the number of epoch, which determines the number of training iterations
n_epoch = 10 

for epoch in range(n_epoch):
  # Set the gradients to 0
  adam.zero_grad()  //每次更新之后都要清空梯度，我们不需要累加梯度

  # Get the model predictions
  y_pred = model(x)  //对模型进行预测

  # Get the loss
  loss = loss_function(y_pred, y)  //算出损失函数

  # Print stats
  print(f"Epoch {epoch}: traing loss: {loss}")

  # Compute the gradients
  loss.backward()  #求偏导数 计算梯度

  # Take a step to optimize the weights
  adam.step()  #更新模型参数
```

```python
Epoch 0: traing loss: 0.5756269693374634
Epoch 1: traing loss: 0.5173695683479309
Epoch 2: traing loss: 0.44215404987335205
Epoch 3: traing loss: 0.34650230407714844
Epoch 4: traing loss: 0.25023093819618225
Epoch 5: traing loss: 0.17002996802330017
Epoch 6: traing loss: 0.1046878844499588
Epoch 7: traing loss: 0.05941229313611984
Epoch 8: traing loss: 0.03189791366457939
Epoch 9: traing loss: 0.016544118523597717
```

可以看到经过十次模型参数的优化，损失之越来越小，真实值和预测值的偏差也越来越小

查看模型参数梯度

```python
# Print the gradients
for name, param in model.named_parameters():
    print(f"Gradient of {name}: {param.grad}")
>>>
Gradient of model.0.weight: tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0271, -0.0100,  0.0015, -0.0144, -0.0061]])
Gradient of model.0.bias: tensor([ 0.0000,  0.0000, -0.0138])
Gradient of model.2.weight: tensor([[ 0.0000,  0.0000, -0.0108],
        [ 0.0000,  0.0000, -0.0108],
        [ 0.0000,  0.0000, -0.0139],
        [ 0.0000,  0.0000, -0.0041],
        [ 0.0000,  0.0000, -0.0050]])
Gradient of model.2.bias: tensor([-0.0038, -0.0038, -0.0048, -0.0018, -0.0021])
```



#### 用训练数据查看模型表现

因为模型使用了sigmoid，在这个例子中，`y_pred` 中的大多数值都非常接近 1，这可能表明模型在训练数据上的表现不错，因为它倾向于对样本做出强烈的正类预测。

```python
# See how our model performs on the training data
y_pred = model(x)
y_pred
>>>
tensor([[0.9987, 0.9998, 0.9983, 0.9993, 0.9993],
        [0.9993, 0.9999, 0.9990, 0.9996, 0.9997],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [0.9946, 0.9988, 0.9932, 0.9959, 0.9964],
        [0.9963, 0.9993, 0.9953, 0.9974, 0.9977],
        [0.9987, 0.9998, 0.9983, 0.9993, 0.9993],
        [0.9999, 1.0000, 0.9998, 1.0000, 1.0000],
        [0.9997, 1.0000, 0.9996, 0.9999, 0.9999],
        [0.9982, 0.9997, 0.9977, 0.9989, 0.9990],
        [0.9997, 1.0000, 0.9996, 0.9999, 0.9999]], grad_fn=<SigmoidBackward>)
```





# Demo

