---
title: "Softmax 与交叉熵"
date: 2026-06-14
draft: false
math: true
tags: ["AI", "math", "deep learning"]
categories: ["AI"]
---

这是一篇用于测试博客基础能力的示例文章：行内公式、块级公式、目录和代码高亮。

给定 logits $z \in \mathbb{R}^K$，softmax 定义为

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}.
$$

对于 one-hot 标签 $y$，交叉熵损失为

$$
\mathcal{L}(z, y)
= - \sum_{i=1}^{K} y_i \log p_i.
$$

如果真实类别是 $c$，上式可以写成

$$
\mathcal{L}(z, c)
= -\log p_c.
$$

代码块示例：

```python
import torch

def softmax(x):
    x = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

## 请早睡早起~
