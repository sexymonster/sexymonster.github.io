---
layout: post
title: "DQN을 이용한 주식 트레이딩(1) - Env설정"
date: 2022-11-06 23:31:29 +0900
categories: Reinforcement
---

# Q-Learning 에이전트 정의

Tensorflow를 활용하여 Agent 클래스를 생성 하려 한다.

```python
def __init__(self, state_size, window_size, trend, skip, batch_size):
    self.state_size = state_size
    self.window_size = window_size
    self.half_window = window = window_size//2
    self.trend = trend
    self.skip = skip
    self.action_size = 3
    self.batch_size = batch_size
    self.memory = deque(maxlen = 1000)
    self.inventory = []
    self.gamma = 0.95
    self.epsilon = 0.5
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.999
    tf.reset_default_graph()

    self.sess = tf.InteractiveSession()
    self.X = tf.placeholder(tf.float32, [None, self.state_size])
    self.Y = tf.placeholder(tf.float32, [None, self.action_size])
    feed = tf.layers.dense(self.X, 256, activation ="relu")  # tf1.에서는 activation=tf.nn.relu

    self.logits = tf.layers.dense(feed, self.action_size)
    self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
    self.optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(self.cost)
    self.sess.run(tf.global_variables_initializer())
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
