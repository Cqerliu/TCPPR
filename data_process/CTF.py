import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# encoder for protein sequence
class ProEncoder:
    elements = 'AIYHRDC'  # 定义一个包含特定氨基酸字符的字符串，在后续编码中作为基础元素使用
    element_number = 7  # elements中包含的元素数量

    # clusters: {A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}
    # 用于将pro_intab中的字符按照对应位置转换为pro_outtab中的字符
    pro_intab = 'AGVILFPYMTSHNQWRKDEC'
    pro_outtab = 'AAAIIIIYYYYHHHHRRDDC'

    # WINDOW_P_UPLIMIT 确定生成k-mer时的窗口大小上限；CODING_FREQUENCY 控制是否对编码向量进行归一化处理；
    # VECTOR_REPETITION_CNN 卷积网络对编码向量中的每个元素进行的重复次数
    def __init__(self, WINDOW_P_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_P_UPLIMIT = WINDOW_P_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN
        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED

        # list and position map for k_mer
        # 每次迭代在已有的k_mer基础上添加elements中的一个字符，逐步生成更长的k_mer，
        # 并将它们添加到k_mer_list中，同时在k_mer_map中记录每个k_mer的位置。
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

        # print(len(self.k_mer_list))
        # print(self.k_mer_list)

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
                else:
                    vec = vec
            result += list(vec)
        return np.array(result)

    def encode_conjoint_cnn(self, seq):  # 进行编码
        result_t = self.encode_conjoint(seq)
        result = np.array([[x] * self.VECTOR_REPETITION_CNN for x in result_t])  # 将其扩展为适合CNN处理的形式
        return result

def read_tsv_file(tsv_file_path):
    sequences = {}
    labels = {}
    with open(tsv_file_path, 'r') as f:
        header = f.readline().strip().split('\t')
        sequence_index = header.index('sequence')
        label_index = header.index('label')
        for line in f:
            line = line.strip().split('\t')
            sequence_name = len(sequences) + 1  # 使用序号作为序列名称
            sequence = line[sequence_index]
            label = line[label_index]
            sequences[sequence_name] = sequence
            labels[sequence_name] = label
    return sequences, labels

def convert_sequences_to_cnn_input(tsv_file_path, encoder):
    sequences, labels = read_tsv_file(tsv_file_path)
    cnn_input_vectors = {}
    for sequence_name, sequence in sequences.items():
        cnn_input_vectors[sequence_name] = encoder.encode_conjoint_cnn(sequence)
    return cnn_input_vectors, labels

if __name__ == "__main__":
    encoder = ProEncoder(WINDOW_P_UPLIMIT=3, CODING_FREQUENCY=True, VECTOR_REPETITION_CNN=1,
                         TRUNCATION_LEN=None, PERIOD_EXTENDED=None)
    tsv_file_path = "../data/coli/polymerase/coli_pro.tsv"
    cnn_input_vectors, labels = convert_sequences_to_cnn_input(tsv_file_path, encoder)
    # 创建一个空的列表，用于存储所有转换后的向量
    all_vectors_list = []
    all_labels = []

    for sequence_name, vector in cnn_input_vectors.items():
        all_vectors_list.append(vector)
        all_labels.append(labels[sequence_name])

    # 将列表转换为numpy数组
    all_vectors_array = np.array(all_vectors_list)
    np.set_printoptions(threshold=np.inf)
    # 输出为.npy文件
    output_file_path = "../data/coli/polymerase/BS1_input_243.npy"
    np.save(output_file_path, all_vectors_array)
    # 加载保存的.npy文件并查看向量内容
    loaded_vectors = np.load(output_file_path)

    print("转换后的向量内容如下：")
    for i, vector in enumerate(loaded_vectors):
        print(f"向量 {i + 1}: {vector}")
    print("维度为：", loaded_vectors.shape)

    # 对标签进行编码
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    print("编码后的标签：", encoded_labels)

    # 保存编码后的标签到.npy文件
    label_output_file_path = "../data/coli/polymerase/labelBS1.npy"
    np.save(label_output_file_path, encoded_labels)
    print(f"编码后的标签已保存到 {label_output_file_path}")