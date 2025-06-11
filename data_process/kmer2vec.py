from gensim.models import word2vec
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

valid_bases = set('ACGT')
# 从 TSV 文件中读取序列数据和标签数据
data = pd.read_csv('../data/HM/promoter/HM1.tsv', sep='\t')
sequences = data['sequence'].values
labels = data['label'].values  # 读取标签数据

kmer = 3  # set the k value

dna2 = [[] for i in range(len(sequences))]

# using the overlapping strategy
for i, seq in enumerate(sequences):
    filtered_seq = ''.join([base for base in seq.upper() if base in valid_bases])
    for j in range(0, len(filtered_seq) - kmer + 1):
        kmer_seq = filtered_seq[j:j + kmer]
        dna2[i].append(kmer_seq)

sentence = []
for i in range(len(sequences)):
    sentence.append(" ".join(dna2[i]))

# 保存处理后的 kmer 序列到文件
with open('../data/HM/promoter/kmer_HM1.txt', 'w') as f:
    for seq in sentence:
        f.write(seq + '\n')

# 设置日志格式，记录每个数据
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 进行 word2vec 训练 sg=1 表示使用跳词模型 vector_size 表示向量维度 epoch 表示设置训练的轮数
sentences = word2vec.Text8Corpus('../data/HM/promoter/kmer_HM1.txt')
model = word2vec.Word2Vec(sentences, sg=1, window=24, min_count=1, negative=5, sample=0.001, hs=0, workers=1, epochs=2,
                          vector_size=100, seed=666)
# 保存模型
model.save('../data/HM/promoter/word2vec_HM.model')
# 重新加载模型
model = word2vec.Word2Vec.load('../data/HM/promoter/word2vec_HM.model')
# 提取了模型中的词向量数据，并将其保存为一个 numpy 数组
x = {}
for word in model.wv.key_to_index.keys():  # check and record all the words and vectors
    x[word] = model.wv[word]
print(x)

np.save('../data/HM/promoter/word2vec_HM1.npy', x)

# 对标签进行编码并建立索引
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 将编码后的标签存储到一个 NumPy 数组中
label_array = np.array(encoded_labels)
# 保存标签数组到.npy 文件
np.save('../data/HM/promoter/HM1_labels.npy', label_array)
print("编码后的标签已保存到 ",label_array)

# 构建标签索引字典
label_index = {label: index for index, label in enumerate(label_encoder.classes_)}
print("标签索引:", label_index)