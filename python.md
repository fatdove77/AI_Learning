# python

## 推导式

https://www.cnblogs.com/cenyu/p/5718410.html

```python
 word2ind = {word: i for i, word in enumerate(words)}
```

- `{word: i for i, word in enumerate(words)}` 是一个字典推导式，它遍历 `enumerate(words)` 生成的每个元组 `(i, word)`，并为每个单词 `word` 创建一个键值对，键是单词 `word`，值是索引 `i`。结果是一个字典，其中每个键是一个唯一单词，每个值是该单词对应的唯一索引。

例如，如果 `words = ['hello', 'world', 'python']`，那么 `word2ind` 将是：

```python
{'hello': 0, 'world': 1, 'python': 2}
```





## 字典的get方法

如果第一个key不存在，就取第二个值作为默认值

```python
word_to_ix.get(token, word_to_ix["<unk>"])
```





## zip函数

在这段代码中，`batch` 是一个列表，其中的每个元素是一个元组 `(x, y)`，`x` 是训练样本（即句子），`y` 是对应的标签（即每个单词是否是地点的标识）。`batch` 的输入格式类似于以下形式：

```python
batch = [
    (["we", "always", "come", "to", "paris"], [0, 0, 0, 0, 1]),
    (["he", "is", "from", "taiwan"], [0, 0, 0, 1])
]
```

`zip(*batch)` 是一个 Python 内置函数，用于将多个列表（或其他可迭代对象）的对应元素打包成元组。在这个例子中，它用于将 `batch` 中的训练样本 (`x`) 和标签 (`y`) 分开，并将它们分别组合成两个列表。使用 `zip(*batch)` 后，`x` 和 `y` 将分别是：

```python
x = [["we", "always", "come", "to", "paris"], ["he", "is", "from", "taiwan"]]
y = [[0, 0, 0, 0, 1], [0, 0, 0, 1]]
```

这样，我们就可以分别对训练样本和标签进行处理。在这个例子中，`x` 将被用于创建输入张量，`y` 将被用于创建标签张量，这些张量将被用于训练模型。

