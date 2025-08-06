import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder


# encoder for protein sequence
class ProEncoder:
    elements = 'AIYHRDC'  # define specific specific specific amino acid characters
    element_number = 7  # number of elements

    # 氨基酸聚类映射
    pro_intab = 'AGVILFPYMTSHNQWRKDEC'
    pro_outtab = 'AAAIIIIYYYYHHHHRRDDC'

    def __init__(self, WINDOW_P_UPLIMIT=3, CODING_FREQUENCY=True, VECTOR_REPETITION_CNN=1,
                 TRUNCMMION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_P_UPLIMIT = WINDOW_P_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN
        self.TRUNCMMION_LEN = TRUNCMMION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        # generate k-mer lists and mappings
        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_P_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i

        self.transtable = str.maketrans(self.pro_intab, self.pro_outtab)

    def encode_conjoint(self, seq):
        seq = seq.translate(self.transtable)
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_P_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                max_value = vec.max()
                if max_value != 0:
                    vec = vec / max_value
            result += list(vec)
        return np.array(result)

    def encode_conjoint_cnn(self, seq):
        result_t = self.encode_conjoint(seq)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])
        cnn_tensor = torch.tensor(result).unsqueeze(0)
        cnn_tensor = cnn_tensor.permute(0, 2, 1)  # convert to [batch, channels, length] format
        return cnn_tensor


# encode for promoter sequences
class PromoterEncoder:
    """
    Generate k-mer sequence vectors of consistent length to be used directly as input to the Transformer.
    """

    def __init__(self, kmer=3, vector_size=100, window=24, epochs=2, seed=666):
        self.kmer = kmer
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.seed = seed
        self.valid_bases = {'A', 'C', 'G', 'T'}

        self.model = None
        self.word_vectors = None
        self.label_encoder = LabelEncoder()
        self.encoded_labels = None
        self.label_index = None
        self.sequences = None
        self.kmers_list = None
        self.sequence_length = None

    def set_sequences_labels(self, sequences, labels):
        self.sequences = sequences
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        self.label_index = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
        return self

    def generate_kmers(self):
        """
        Generate k-mers for all sequences
        """
        if self.sequences is None:
            raise ValueError("请先通过set_sequences_labels设置序列")

        self.kmers_list = [self._preprocess_single(seq) for seq in self.sequences]
        # check if all k-mer sequences have the same length.
        lengths = [len(kmers) for kmers in self.kmers_list]
        if len(set(lengths)) != 1:
            raise ValueError("所有序列的k-mer长度必须一致")
        self.sequence_length = lengths[0]
        return self

    def _preprocess_single(self, seq):
        """
        Generate k-mers for single sequences
        """
        filtered = ''.join([b for b in seq.upper() if b in self.valid_bases])
        return [filtered[j:j + self.kmer] for j in range(len(filtered) - self.kmer + 1)]

    def train_word2vec(self):
        """
        train Word2Vec model
        """
        if self.kmers_list is None:
            raise ValueError("请先生成k-mer序列")

        self.model = Word2Vec(
            sentences=self.kmers_list,
            sg=1, window=self.window, min_count=1,
            negative=5, sample=0.001, hs=0, workers=1,
            epochs=self.epochs, vector_size=self.vector_size, seed=self.seed
        )
        self.word_vectors = {word: self.model.wv[word] for word in self.model.wv.key_to_index}
        return self

    def get_transformer_input(self):
        """
        generate Transformer input
        """
        if self.word_vectors is None:
            raise ValueError("请先训练Word2Vec模型")

        X = []
        for kmers in self.kmers_list:
            vecs = [self.word_vectors[kmer] for kmer in kmers if kmer in self.word_vectors]
            if len(vecs) != self.sequence_length:
                raise ValueError(f"k-mer向量转换后长度异常：{len(vecs)}（预期：{self.sequence_length}）")
            X.append(np.array(vecs, dtype=np.float32))

        return np.array(X), self.encoded_labels

    def run_pipeline(self, sequences, labels):
        self.set_sequences_labels(sequences, labels) \
            .generate_kmers() \
            .train_word2vec()
        return self.get_transformer_input()