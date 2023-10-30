import torch
import torch.utils.data as tud
from dataset import WordEmbeddingDataset
from model import EmbeddingModel
from train import trainer

from collections import Counter
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()
# 为了保证实验结果可以复现，经常会把各种random seed固定在一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

# 超参数设置
NUM_EPOCHS = 5  # 训练轮数
MAX_VOCAB_SIZE = 30000  # 词汇大小
BATCH_SIZE = 128  # 批量大小
LEARNING_RATE = 0.2  # 学习率
EMBEDDING_SIZE = 100


# tokenize函数，将一篇文本转化为一个个单词
def word_tokenize(text):
    return text.split()


"""
1.通过文本文件中读取的单词来创建vocabulary
2.由于单词数量可能太大，只选取最常见的MAX_VOCAB_SIZE个单词
3.添加一个UNK表示所有不常见的单词
4.需要记录 word_to_id, id_to_word, 每个单词的出现频率
"""
with open("./data/text/text.train.txt", "r") as f:
    text = f.read()

text = [w for w in word_tokenize(text.lower())]
vocab = dict((Counter(text).most_common(MAX_VOCAB_SIZE - 1)))  # 字典格式
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
VOCAB_SIZE = len(idx_to_word)

# 统计每个单词的频率，根据论文里的公式来乘以3/4次方，用来做negative sampling
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
# 重新归一化
word_freqs = word_freqs / np.sum(word_freqs)

print("====预处理完成====")

# 创建dataset和dataloader
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs)
print("====dataset构建完成====")
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(next(iter(dataloader))[0].shape)  # 一个batch中间词维度 torch.Size([128])
print(next(iter(dataloader))[1].shape)  # 一个batch周围词维度 torch.Size([128, 6])
print(next(iter(dataloader))[2].shape)  # 一个batch负样本维度 torch.Size([128, 60])

# 定义模型并把模型放到GPU上
model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
print("====模型完成====")

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 训练
trainer(model, optimizer, NUM_EPOCHS, dataloader, word_to_idx, idx_to_word, USE_CUDA)
