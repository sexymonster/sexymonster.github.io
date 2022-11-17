---
layout: post
title: "DQN을 이용한 주식 트레이딩(1)"
date: 2022-11-06 23:31:29 +0900
categories: Reinforcement
---

강화학습을 공부하다가 DQN을 이용해서 주식 트레이딩 알고리즘을 만들고 싶었습니다. 하지만 아직 강화학습을 위한 환경 설정, 에이전트 정의 등 미숙한 부분이 너무 많아요... 그래서 아주 기초적인 부분부터 차근차근 만들어 보려고 합니다.

# 알고리즘 계획

주식정보는 일별 종가로 이루어져 있습니다.\
state는 현시점에서 전날의 종가 대비 당일 종가의 증가량으로 구성하였고, action은 유지, 매수, 매도로 3가지 경우의 수가 있습니다. \
매수 신호가 오면 주식을 1주만 사고, 매도 신호가 왔을 때 그 주식을 판매합니다. 이때 차익을 최대로 하는 값을 reward로 지정한다.

# 버전 정보

python = 3.9.12\
tensorflow = 2.10.0\
numpy = 1.21.5\
pandas = 1.4.2\
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

여기서 state를 나타내는 리스트를 반환하는 함수를 생성하려고 합니다.\
data = 주식의 종가들을 나타내는 시리즈\
timestep = 임의의(원하는) 시점\
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

강화학습을 통해 스스로 학습하는 Agent를 클래스 형태로 생성하려 합니다.

여기서 state_size는 위에서 지정했던 window_size와 같습니다. 그리고 action_space는 유지, 매수, 매도로 3가지 action이 있었기 때문에 여기서는 3입니다. \
Agent는 기본적으로 Q함수의 값이 높은 행동을 선택하지만, 이로 인해 다양한 해결책을 탐색하지 않고 적당히 좋은 해결책을 찾아내면 그것에 수렴해버리는 문제가 생기는 것을 방지하기 위해 e-greedy 방법이라는 탐색법을 사용합니다. 처음 epsilon의 값은 1이고, episode가 끝날때 마다 0.995를 곱합니다.

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

0~1 사이의 랜덤한 수가 epsilon보다 작을 때 action은 무작위로 진행됩니다. epsilon은 1부터 시작해 조금씩 줄어들기 때문에 후반으로 갈수록 적당히 좋은 해결책의 action이 진행됩니다.\
여기서 act_values는 이중 리스트로 구성되어있기 때문에 act_values[0]으로 나타내줘야 합니다.

```python
    def trade(self, state):
        if random.random() <= self.epsilon:
        return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
```

Keras의 Sequential 모델을 이용해 모델링을 해줍니다.\
입력 차원은 state_size이고, 출력 차원은 action_space입니다. 손실함수는 일단 mse로 하였고 Adam 옵티마이저를 사용하였습니다.

```python
    def model_builder(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer="Adam")

        return model
```

![alt text](/public/img/DQNtrader_model.png)

memory에 저장된 값들을 일정량 만큼 batch에 따로 빼놓고 이용합니다.
미래 가치에 감가율을 적용한 Q함수는 아래와 같습니다.\
![alt text](/public/img/Q함수.png)\
위와 같은 공식을 이용해 reward를 구해줍니다.\
epsilon값이 epsilon_final보다 크다면 epsilon_decay만큼 곱해서 epsilon값의 크기를 줄여줍니다.

```python
    def batch_train(self, batch_size):

        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state)

        if not done:
            target[0][action] = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        target[0][action] = reward

        self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
```

여기까지가 Agent 클래스를 정의한 코드입니다. 이제 본격적으로 학습을 시켜보겠습니다.

미국의 apple주식의 종가를 이용합니다. 데이터가 많으면 학습률을 높일 수 있지만 학습하는데 시간이 오래 걸리기 때문에 여기서는 최근 50개의 데이터를 이용합니다. GPU를 보유하고 있고, 학습률을 올리고 싶다면 데이터의 개수를 늘려도 좋을거 같습니다.

```python
stock_name = "AAPL"
data = dataset_loader(stock_name).iloc[-50:]
data
```

### 파라미터

window_size = 5\
episodes = 100\

batch_size = 8\
data_samples = len(data) - 1\

# 실행 코드

total_pro : 각 episode에서 구한 profit을 저장하는 리스트\
start_time, end_time : 학습을 시작하는 시점과 학습을 종료하는 시점에서의 시각, 학습에 소요된 시간을 나타내기위해 이용됨\
episode가 10번 진행될때마다 모델 저장\
특별히 어려운 내용은 없어서 코드를 읽어보면 흐름이 이해 될거라 생각합니다.

```python
total_pro = []
start_time = datetime.datetime.now()
for episode in range(1, episodes + 1):

    print("Episode: {}/{}".format(episode, episodes))

    state = state_creator(data, 10, window_size + 1)

    action = [0]
    total_profit = 0
    trader.inventory = []

    # for t in tqdm(range(data_samples)):
    for t in tqdm(range(window_size,data_samples)):

        action = trader.trade(state)

        next_state = state_creator(data, t+1, window_size + 1)
        reward = 0

        if action == 1: #Buying
            trader.inventory.append(data[t])
            print("AI Trader bought: ", stocks_price_format(data[t]))

        elif action == 2 and len(trader.inventory) > 0: #Selling
            buy_price = trader.inventory.pop(0)

            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print("AI Trader sold: ", stocks_price_format(data[t]), " Profit: " + stocks_price_format(data[t] - buy_price) )

        if t == data_samples - 1:
            done = True
        else:
            done = False

        trader.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("########################")
            total_pro.append(total_profit)
        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    if episode % 10 == 0:
        trader.model.save("basic_ai_trader_{}.h5".format(episode))
end_time = datetime.datetime.now()
print("총 소요 시간: {}".format(end_time - start_time))
```

이대로 진행하면 아래와 같은 결과가 나왔습니다.\
![alt text](/public/img/basic_trader_result.png)\
이득과 손해가 최대 30달러 정도임을 알 수 있습니다. episode들의 profit 평균은 5.4달러 정도입니다. 주식을 한주씩 거래했기 때문에 profit의 범위가 30달러 정도인거 같습니다.
