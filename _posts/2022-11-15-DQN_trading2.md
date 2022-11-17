---
layout: post
title: "DQN을 이용한 주식 트레이딩(2)"
date: 2022-11-15 23:31:29 +0900
categories: Reinforcement
---

지난 시간에 간단한 주식 트레이딩 알고리즘을 만들어 보았습니다.\
하지만 1주씩 매수/매도하는 것은 일반적으로 우리가 주식 거래하는 모습과 매우 다르고 많은 수익을 기대하기도 어렵습니다. 그래서 거래 방식을 조금 보완하고 추가로 시각화도 가능하게 만들려고 합니다.

# 업데이트 계획

함수들 중에 필요한 기능들이 있으면 추가합니다. \
자본금의 개념을 도입하여 1주씩 거래하던 방식이 아닌 보유한 현금내에서 적정량의 주식을 거래하는 방식으로 변경합니다.\
거래내역들을 기록하여 시각화를 합니다.

# 함수 업데이트

dataset_loader 함수에서 시작일과 종료일을 출력하여 데이터 셋의 범위를 알려주도록 업데이트하였습니다.

```python
def dataset_loader(stock_name):

    dataset = data_reader.DataReader(stock_name, data_source="yahoo")

    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]

    close = dataset['Close']
    print("시작일: {}, 종료일: {}".format(start_date, end_date))
    return close
```

# 실행 코드 업데이트

매수 또는 매도를 좌표 평면에 나타낼때 거래량도 함께 나타내기 위해 amount(거래량을 기록하는 리스트)를 추가 하였습니다. amount에 대한 코드만 나타내보면 아래와 같습니다. 매수할때 매수한 주식의 수를 저장, 매도할때 매도한 주식의 수를 저장, 유지할때는 0을 저장하였습니다.

```python
for episode in range(1, episodes + 1):

    amount = [] # 거래량

    for t in tqdm(range(window_size,data_samples)):
        if action == 1 and cash > data[t]: #Buying
            amount.append(num_stock)
        elif action == 2 and len(trader.inventory) > 0: #Selling
            amount.append(buy_stock[1])
        else:
            amount.append(0)
```

자본금을 가지고 구매할 수 있는 주식의 30퍼센트만 매수하도록 한다. 풀매수를 하지 않는 이유는, 풀매수를 한 다음 스텝에서 다시 매수 타이밍이 왔을 때 아무런 action을 할 수 없어서 임의로 30퍼센트라는 값을 부여했습니다. \
그리고 거래 과정을 도표로 나타내기 위해 sells, buys, closes 리스트도 추가하였습니다. 여기서 주의할 점은 매수 시점에서는 sells 리스트에만 주가를 추가 하는게 아니라, buys리스트에도 None 값을 반드시 추가 해줘야 한다는 점입니다. 그렇지 않으면 리스트들의 x값이 일치하지 않아 플롯할 수 없습니다.

```python
assets = []
start_time = datetime.datetime.now()

for episode in range(1, episodes + 1):

    cash = capital # 자본금
    trader.inventory = []
    sells = []
    buys = []
    closes = []

    # for t in tqdm(range(data_samples)):
    for t in tqdm(range(window_size,data_samples)):
        closes.append(data[t])

        if action == 1 and cash > data[t]: #Buying
            stock = divmod(cash, data[t])

            if stock[0]*0.3 > 1:
                num_stock = math.trunc(stock[0]*0.3)
                cash -= data[t]*float(num_stock)
                print("purchase price: {0:.2f} $, amount of purchase: {1}, cash: {2:.2f} $".format(data[t],num_stock,cash))

            else:
                num_stock = int(stock[0])
                cash -= data[t]*num_stock
                print("purchase price: {0:.2f} $, amount of purchase: {1}, cash: {2:.2f} $".format(data[t],num_stock,cash))

            trader.inventory.append((data[t],num_stock))
            buys.append(data[t])
            sells.append(None)



        elif action == 2 and len(trader.inventory) > 0: #Selling
            buy_stock = trader.inventory.pop(0)

            reward += max((data[t] - buy_stock[0])*buy_stock[1], 0)
            cash += data[t]*buy_stock[1]

            sells.append(data[t])
            buys.append(None)
            print("selling price: {0:.2f} $, amount of selling: {1}, cash: {2:.2f} $".format(data[t],int(buy_stock[1]),cash))

        else:
            buys.append(None)
            sells.append(None)
        if t == data_samples - 1:
            done = True
        else:
            done = False

        trader.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            if len(trader.inventory) == 0:
                print("########################")
                print("TOTAL ASSET: {:.2f} $".format(cash))
                print("########################")
                assets.append(cash)
            else:
                num_stocks = 0
                for a in trader.inventory:
                    num_stocks += a[1]
                asset = num_stocks * data[-1] + cash
                print("########################")
                print("TOTAL ASSET: {:.2f} $".format(asset))
                print("########################")
                assets.append(asset)
        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)

    if episode % 10 == 0:
        trader.model.save("ai_trader_{}.h5".format(episode))
end_time = datetime.datetime.now()
print("총 소요 시간: {}".format(end_time - start_time))
```

위의 코드에서 이전의 amount에 대한 코드를 추가하면 아래와 같은 결과가 나옵니다.\
![alt text](/public/img/DQNtrader_update_result.png)\
그리고 해당 거래에서는 최종적으로 377달러의 수익을 냈습니다.
