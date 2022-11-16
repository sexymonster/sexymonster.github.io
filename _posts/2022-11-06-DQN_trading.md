---
layout: post
title: "DQN을 이용한 주식 트레이딩(1)"
date: 2022-11-06 23:31:29 +0900
categories: Reinforcement
---

강화학습을 공부하다가 DQN을 이용해서 주식 트레이딩 알고리즘을 만들고 싶었습니다. 하지만 아직 강화학습을 위한 환경 설정, 에이전트 정의 등 미숙한 부분이 너무 많아요... 그래서 아주 기초적인 부분부터 차근차근 만들어 보려고 합니다.

# 알고리즘 계획

주식정보는 일별 종가로 이루어져 있습니다. state는 현시점에서 전날의 종가 대비 당일 종가의 증가량으로 구성하였고, action은 유지, 매수, 매도로 3가지 경우의 수가 있습니다. \
매수 신호가 오면 주식을 1주만 사고, 매도 신호가 왔을 때 그 주식을 판매합니다. 이때 차익을 최대로 하는 값을 reward로 지정한다.

# 버전 정보

python = 3.9.12
tensorflow = 2.10.0
numpy = 1.21.5
pandas = 1.4.2
pandas-datareader = 0.10.0

# 패키지 로드

```python
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
import datetime

from tqdm import tqdm_notebook, tqdm
from collections import deque

```

```python

```

# 각종 함수 정의

당일 종가가 전일 종가에 비해 얼마나 증가했는지를 0~1사이의 수로 나타내기 위해서 모든 점에서 음이 아닌 미분값을 가지는 시그모이드 함수를 정의

```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
```

매수 또는 매도한 주식의 가격, 그리고 이익을 소숫점 두자리 까지 나타내기 위한 함수

```python
def stocks_price_format(n):
    if n < 0:
        return "- $ {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))
```

주식 데이터를 불러오는 함수. 이번 프로젝트에서는 일별 종가를 반환함

```python
def dataset_loader(stock_name):

    dataset = data_reader.DataReader(stock_name, data_source="yahoo")

    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[1]).split()[0]

    close = dataset['Close']

    return close
```

여기서 state를 나타내는 리스트를 반환하는 함수를 생성하려고 합니다.
data = 주식의 종가들을 나타내는 시리즈
timestep = 임의의(원하는) 시점
window_size = 관찰해야 하는 정보의 step수

```python
def state_creator(data, timestep, window_size):

    starting_id = timestep - window_size + 1
    if starting_id >= 0:
        windowed_data = data[starting_id:timestep+1]
    else:
        windowed_data = starting_id * [data[0]] + list(data[0:timestep+1])

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))

    return np.array([state])
```

# 에이전트 정의

Tensorflow를 활용하여 Agent 클래스를 생성 하려 한다.

```python
class AI_Trader():

    def __init__(self, state_size, action_space=3, model_name="AITrader"):

        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen = 2000)
        self.inventory = []
        self.model = self.model_builder()
        self.model_name = self.model.name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
```

```python
def trade(self, state):
      if random.random() <= self.epsilon:
          return random.randrange(self.action_space)

      actions = self.model.predict(actions[0])
```

```python
def trade(self, state):
    if random.random() <= self.epsilon:
      return random.randrange(self.action_space)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])
```
