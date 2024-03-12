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

