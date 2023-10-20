from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

input_file = 'word2vec/wiki.txt'
out_file = 'word2vec/wiki.model'

'''
LineSentence
size 词向量的维度
window 窗口大小
min_count 单词最少出现的次数
sg 值为1是skip-gram，0为cbow
hs 值为0采用negative sampling
'''
model = Word2Vec(LineSentence(input_file),
                 vector_size=128,
                 window=5,
                 min_count=5,
                 workers=multiprocessing.cpu_count(),
                 sg=1,
                 hs=0,
                 negative=10)

model.save(out_file)
