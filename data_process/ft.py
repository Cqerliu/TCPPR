import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gensim.models import word2vec
import torch
import torch.nn as nn
from tqdm import tqdm
from transformer import PrompterTransformer


class FeatureFT:
    def __init__(self, kmer=3, sg=1, window=24, min_count=1, negative=5, sample=0.001, hs=0, workers=1, epochs=2,
                 vector_size=100, seed=666, input_dim=100, embedding_dim=64, output_dim=64, depth=8, heads=12,
                 dim_head=64, attn_dropout=0.2, ff_dropout=0.1, batch_size=32):
        self.kmer = kmer
        self.sg = sg
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.sample = sample
        self.hs = hs
        self.workers = workers
        self.epochs = epochs
        self.vector_size = vector_size
        self.seed = seed
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.batch_size = batch_size
        self.valid_bases = set('ACGT')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def read_tsv_file(self, tsv_file_path):
        data = pd.read_csv(tsv_file_path, sep='\t')
        sequences = data['sequence'].values
        labels = data['label'].values
        return sequences, labels

    def process_sequences(self, sequences):
        all_kmers = []
        for seq in tqdm(sequences, desc="Processing sequences"):
            filtered_seq = ''.join([base for base in seq.upper() if base in self.valid_bases])
            kmers = []
            for j in range(0, len(filtered_seq) - self.kmer + 1):
                kmer_seq = filtered_seq[j:j + self.kmer]
                kmers.append(kmer_seq)
            all_kmers.append(kmers)
        return all_kmers

    def test_word2vec(self, all_kmers, save_path_model):
        # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = word2vec.Word2Vec(all_kmers, sg=self.sg, window=self.window, min_count=self.min_count,
                                  negative=self.negative, sample=self.sample, hs=self.hs, workers=self.workers,
                                  epochs=self.epochs, vector_size=self.vector_size, seed=self.seed)
        model.save(save_path_model)
        x = {}
        for word in tqdm(model.wv.key_to_index.keys(), desc="Extracting word vectors"):
            x[word] = model.wv[word]
        return x

    def encode_labels(self, labels):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        label_tensor = torch.from_numpy(np.array(encoded_labels)).long().to(self.device)
        label_index = {label: index for index, label in enumerate(label_encoder.classes_)}
        return label_tensor, label_index

    def initialize_embedding_layer(self, loaded_word2vec_dict):
        vocab_size = len(loaded_word2vec_dict)
        word_to_index = {word: i for i, word in enumerate(loaded_word2vec_dict.keys())}
        index_to_word = {i: word for word, i in word_to_index.items()}
        embedding_dim = list(loaded_word2vec_dict.values())[0].shape[0]
        embedding_layer = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        for i, word in enumerate(loaded_word2vec_dict.keys()):
            np_array = loaded_word2vec_dict[word].copy()
            embedding_layer.weight.data[i] = torch.from_numpy(np_array).to(self.device)
        return embedding_layer, word_to_index, index_to_word

    def convert_sequences_to_indices(self, all_kmers, word_to_index):
        index_sequences = []
        lengths = []
        for kmers in tqdm(all_kmers, desc="Converting sequences to indices"):
            index_sequence = []
            for mer in kmers:
                if mer in word_to_index:
                    index_sequence.append(word_to_index[mer])
            index_sequences.append(index_sequence)
            lengths.append(len(index_sequence))
        return index_sequences

    def extract_features(self, input_sequences):
        all_kmers = self.process_sequences(input_sequences)
        loaded_word2vec_dict = self.test_word2vec(all_kmers, '../data/AT/promoter/word2vec_AT.npy')
        label_tensor, _ = self.encode_labels(self.labels)

        embedding_layer, word_to_index, _ = self.initialize_embedding_layer(loaded_word2vec_dict)

        model = PrompterTransformer(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout).to(self.device)
        model.embedding_layer = embedding_layer

        index_sequences = self.convert_sequences_to_indices(all_kmers, word_to_index)

        all_outputs = []
        model.eval()
        with torch.no_grad():
            num_batches = len(index_sequences) // self.batch_size + (
                1 if len(index_sequences) % self.batch_size != 0 else 0)
            for i in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(index_sequences))
                batch = index_sequences[start_idx:end_idx]
                batch_tensor = torch.tensor(batch, device=self.device)
                outputs = model(batch_tensor)
                all_outputs.append(outputs.cpu().numpy())
                del batch_tensor, outputs
                torch.cuda.empty_cache()

        outputs_np = np.concatenate(all_outputs, axis=0)
        features_tensor = torch.from_numpy(outputs_np).float().to(self.device)
        print("FT的维度为：", features_tensor.shape)
        return features_tensor, label_tensor

    def process(self, tsv_file_path):
        sequences, self.labels = self.read_tsv_file(tsv_file_path)
        features, labels = self.extract_features(sequences)
        return features, labels

if __name__ == "__main__":
    feature_extractor = FeatureFT()
    tsv_file_path = "../data/AT/promoter/AT.tsv"
    features, labels = feature_extractor.process(tsv_file_path)

    # 将特征向量保存为.npy文件
    features_np = features.cpu().numpy()  # 将特征向量从GPU移到CPU并转换为numpy数组
    np.save('../data/AT/promoter/FT_AT.npy', features_np)

    # 将标签保存为.npy文件
    labels_np = labels.cpu().numpy()  # 将标签从GPU移到CPU并转换为numpy数组
    np.save('../data/AT/promoter/AT_labels.npy', labels_np)

    print("特征向量和标签已成功保存为.npy文件。")