import torch
from torch import nn
import jieba
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from utils import fix_seed


def tokenize(text):
    segment = list(jieba.cut(text, cut_all=False))
    return segment


# 将输入的文本序列转换为词典中的 index
def seq2index(seq, vocab):
    segment = tokenize(seq)
    seg_index = []
    for seg in segment:
        seg_index.append(vocab.get(seg, 1))  # 若词不在词典中，则标记为 [UNK]
    return seg_index


# 设置最大长度，如果小于最大长度则进行填充
def padding_seq(seq_list, max_len=20):
    return np.array([np.concatenate([seq, [0] * (max_len - len(seq))]) if len(seq) < max_len else seq[:max_len] for seq in seq_list])


def load_data(batch_size=32):
    train_text = []
    train_label = []
    with open('../data/sentiment/sentiment.train.data', encoding='utf-8') as file:
        for line in file.readlines():
            text, label = line.strip().split('\t')
            train_text.append(text)
            train_label.append(int(label))

    dev_text = []
    dev_label = []
    with open('../data/sentiment/sentiment.valid.data', encoding='utf-8') as file:
        for line in file.readlines():
            text, label = line.strip().split('\t')
            dev_text.append(text)
            dev_label.append(int(label))

    # 使用train_text构造词典
    segment = [tokenize(text) for text in train_text]

    word_frequency = defaultdict(int)
    for row in segment:
        for word in row:
            word_frequency[word] += 1
    # 根据词频降序排序
    word_sort = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

    # 构造词典vocab
    vocab = {'[PAD]': 0, '[UNK]': 1}
    for word in word_sort:
        vocab[word[0]] = len(vocab)

    # 构造训练数据
    train_x = padding_seq([seq2index(seq, vocab) for seq in train_text])
    train_y = np.array(train_label)
    train_dataset = TensorDataset(torch.from_numpy(train_x),
                                  torch.from_numpy(train_y))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    # 构造验证数据
    dev_x = padding_seq([seq2index(seq, vocab) for seq in dev_text])
    dev_y = np.array(dev_label)
    dev_dataset = TensorDataset(torch.from_numpy(dev_x),
                                torch.from_numpy(dev_y))
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size)

    return train_dataloader, dev_dataloader, vocab


# 训练模型
class LstmCLS(torch.nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size):
        super(LstmCLS, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # [batch_size, seq_len, hidden_size]
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)
        self.dense1 = nn.Linear(256, 100)
        self.dense2 = nn.Linear(100, 2)

    def forward(self, x):
        embedding = self.embedding(x)
        out, _ = self.lstm(embedding)  # [batch_size, seq_len, hidden_size]
        out = self.dense1(out[:, -1, :])  # 这里是取序列中的最后一位 [batch_size, hidden_size]
        out = self.dense2(out)
        return out


def train():
    fix_seed()

    train_dataloader, dev_dataloader, vocab = load_data(128)
    model = LstmCLS(vocab_size=len(vocab),
                    embedding_size=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(10):
        print('epoch: ', epoch + 1)

        train_pred = []
        train_label = []
        for step, (b_x, b_y) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                b_x = b_x.cuda().long()
                b_y = b_y.cuda().long()
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 预测标签
            train_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            # 真实标签
            train_label.extend(b_y.cpu().numpy())

        train_acc = accuracy_score(train_pred, train_label)
        print('train acc: ====>', train_acc)

        # 每训练完一轮进行一次验证
        dev_pred = []
        dev_label = []
        for step, (b_x, b_y) in enumerate(dev_dataloader):
            if torch.cuda.is_available():
                b_x = b_x.cuda().long()
                # b_y = b_y.cuda().long()
            with torch.no_grad():
                output = model(b_x)
            dev_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            dev_label.extend(b_y.numpy())
        dev_acc = accuracy_score(dev_pred, dev_label)
        print('dev acc: ====>', dev_acc)
        print()


if __name__ == '__main__':
    train()
