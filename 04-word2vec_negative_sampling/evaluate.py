from model import EmbeddingModel
import torch
from train import evaluate, find_nearest
from collections import Counter
import numpy as np
import scipy

EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 30000
VOCAB_SIZE = 30000
USE_CUDA = torch.cuda.is_available()


# tokenize函数，将一篇文本转化为一个个单词
def word_tokenize(text):
    return text.split()


with open("./data/text/text.train.txt", "r") as f:
    text = f.read()

text = [w for w in word_tokenize(text.lower())]
vocab = dict((Counter(text).most_common(MAX_VOCAB_SIZE - 1)))  # 字典格式
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

# 定义模型并把模型放到GPU上
model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))

embedding_weights = model.input_embeddings()
print("simlex-999", evaluate("./data/simlex-999.txt", embedding_weights, word_to_idx))
print("men", evaluate("./data/men.txt", embedding_weights, word_to_idx))
print("wordsim353", evaluate("./data/wordsim353.csv", embedding_weights, word_to_idx))

# 寻找nearest neighbors
for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
    print(word, find_nearest(word, embedding_weights, word_to_idx, idx_to_word))

# 单词之间的关系
man_idx = word_to_idx["man"]
king_idx = word_to_idx["king"]
woman_idx = word_to_idx["woman"]
embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
for i in cos_dis.argsort()[:20]:
    print(idx_to_word[i])
