"""
FizzBuzz是一个简单的小游戏。游戏规则如下：从1开始往上数数，当遇到3的倍数的时候，说fizz;
当遇到5的倍数，说buzz;当遇到15的倍数，就说fizzbuzz，其他情况下则正常数数。
"""
import numpy as np
import torch


NUM_DIGITS = 10


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


# 训练数据
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# 定义模型
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)
# 定义损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 训练
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每训练完一轮打印loss
    loss = loss_fn(model(trX), trY).item()
    if epoch % 500 == 0:
        print('Epoch:', epoch, 'Loss:', loss)

# 测试数据
testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
with torch.no_grad():
    testY = model(testX)

predictions = zip(range(1, 101), testY.max(1)[1].data.tolist())  # max(1)[0]表示取的最大数值，max(1)[1]表示最大数值的索引

print([fizz_buzz_decode(i, x) for (i, x) in predictions])
print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1, 101)])))
