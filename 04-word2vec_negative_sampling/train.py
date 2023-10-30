"""
定义训练和评估函数
"""
import torch
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity

LOG_FILE = "word-embedding.log"
EMBEDDING_SIZE = 100


def evaluate(filename, embedding_weights, word_to_idx):
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:
        data = pd.read_csv(filename, sep="\t")
    human_similarity = []
    model_similarity = []
    for i in data.index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))

    return scipy.stats.spearmanr(human_similarity, model_similarity)


def find_nearest(word, embedding_weights, word_to_idx, idx_to_word):
    idx = word_to_idx[word]
    embedding = embedding_weights[idx]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]


def trainer(model, optimizer, epochs, dataloader, word_to_idx, idx_to_word, USE_CUDA):
    for epoch in range(epochs):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            if USE_CUDA:
                input_labels = input_labels.cuda()
                pos_labels = pos_labels.cuda()
                neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with open(LOG_FILE, 'a') as f:
                    f.write("epoch:{}, iter:{}, loss:{}\n".format(epoch, i, loss.item()))
                    print("epoch:{}, iter:{}, loss:{}\n".format(epoch, i, loss.item()))

            if i % 2000 == 0:
                embedding_weights = model.input_embeddings()
                sim_simlex = evaluate("./data/simlex-999.txt", embedding_weights, word_to_idx)
                sim_men = evaluate("./data/men.txt", embedding_weights, word_to_idx)
                sim_353 = evaluate("./data/wordsim353.csv", embedding_weights, word_to_idx)

                with open(LOG_FILE, "a") as f:
                    print("epoch:{}, iteration:{}, simlex-999:{}, men:{}, sim353:{}, nearest to monster:{}\n".format(
                        epoch, i, sim_simlex, sim_men, sim_353, find_nearest("monster", embedding_weights, word_to_idx, idx_to_word)))
                    f.write("epoch:{}, iteration:{}, simlex-999:{}, men:{}, sim353:{}, nearest to monster:{}\n".format(
                        epoch, i, sim_simlex, sim_men, sim_353, find_nearest("monster", embedding_weights, word_to_idx, idx_to_word)))

        # 每轮训练完后保存embedding权重
        embedding_weights = model.input_embeddings()
        np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
        torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))
