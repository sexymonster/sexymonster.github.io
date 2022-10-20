---
layout: post
title: "Error 사전"
categories: Python
---

# Type Error

## TypeError: bad operand type for unary -: 'str'

문자열의 연산은 "+", "\*"만 가능, "-"는 사용하지 못한다

```python
if word[i] in spell:
    word =- word[i]  # error
```

```python
if word[i] in spell:
    word = word.strip(word[i])  # modified
    # strip 은 중복된 해당 문자도 전부 삭제함에 주의!
```
