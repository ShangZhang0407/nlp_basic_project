"""
实现Dataloade，需要以下内容：
1.把所有text编码成数字，然后用subsampling预处理这些文字
2.保存vocabulary，单词count，normalized word frequency
3.每个iteration需要sample一个中心词
4.根据当前的中心词返回context单词
5.根据当前的中心词sample一些negative单词
6.返回单词的counts

为了适用dataloader，我们需要定义以下两个function:
    __len__: 需要返回整个数据集中有多少个item
    __getitem__: 根据给定的index返回一个item
"""
import torch
import torch.utils.data as tud

# 超参数设置
K = 100  # 负采样样本数
C = 3  # 窗口大小
VOCAB_SIZE = 30000


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs):
        """
            text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        """
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]  # 将文本编码成数字
        self.text_encoded = torch.LongTensor(self.text_encoded)  # 转换成longtensor类型

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)  # 转换成tensor类型

    def __len__(self):
        """
        :return: 返回整个数据集（所有单词）的长度
        """
        return len(self.text_encoded)

    def __getitem__(self, idx):
        """
        :param idx: 单词索引
        :return: 返回中心词、中心词的context单词作为positive单词、随机采样的K个单词作为negative sample单词
        """
        center_word = self.text_encoded[idx]  # 中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 上下文单词索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 超出边界时进行取余操作
        pos_words = self.text_encoded[pos_indices]  # 上下文单词的编码，即positive word

        # 负例采样单词，torch.multinomial作用是按照self.word_freqs的概率做K * pos_words.shape[0]次取值，
        # 输出的是self.word_freqs对应的下标。取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大。
        # 每个正确的单词采样K个，pos_words.shape[0]是正确单词数量=6
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words
