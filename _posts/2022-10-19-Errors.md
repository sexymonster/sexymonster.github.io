---
layout: post
title: "Error 사전"
date: 2022-10-19 23:31:29 +0900
categories: Python
---

# Type Error

## TypeError: bad operand type for unary -: 'str'

문자열의 연산은 "+", "\*"만 가능, "-"는 사용하지 못한다.

```python
if word[i] in spell:
    word =- word[i]  # error
```

```python
if word[i] in spell:
    word = word.strip(word[i])  # modified
    # strip 은 중복된 해당 문자도 전부 삭제함에 주의!
```

# Syntax Error

## SyntaxError: invalid character in identifier

코드를 복사 붙여넣기 하면 가끔 이런 오류가 생긴다.
그냥 다시 같은 문장을 직접 작성하면 오류가 사라진다.
