"""
定义pytorch模型EmbeddingModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size  # 30000
        self.embed_size = embed_size  # 100

        initrange = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)  # 权重初始化的一种方法

        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        """
            input_labels: 中心词 [batch_size]
            pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
            neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]

            return: loss [batch_size]
        """
        input_embedding = self.in_embed(input_labels)  # (batch_size, embed_size)
        pos_embedding = self.out_embed(pos_labels)  # (batch_size, window_size*2, embed_size)
        neg_embedding = self.out_embed(neg_labels)  # (batch_size, window_size*2*K, embed_size)

        # torch.bmm() 表示batch间的矩阵相乘 (b, n, m) * (b, m, p) = (b, n, p)
        # (batch_size, window_size*2, embed_size) * (batch_size, embed_size, 1) -> (batch_size, window_size*2)
        pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()
        # (batch_size, window_size*2*K, embed_size) * (batch_size, embed_size, 1) -> (batch_size, window_size*2*K)
        neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # 注意这里有个负号

        # 根据论文里的公式计算loss
        log_pos = F.logsigmoid(pos).sum(1)  # (batch_size)
        log_neg = F.logsigmoid(neg).sum(1)  # (batch_size)
        loss = log_pos + log_neg  # 优化目标：对数似然函数的值越大越好

        return -loss

    def input_embeddings(self):
        """
        取出self.in_embed的权重参数，维度为(30000, 100)，即得到训练的词向量
        模型训练了两个矩阵：self.in_embed和self.out_embed，但作者认为 self.in_embed比较好，所以舍弃了 out_embed
        :return:
        """
        return self.in_embed.weight.data.cpu().numpy()
