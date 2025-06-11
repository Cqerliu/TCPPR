import numpy as np
from sklearn.preprocessing import LabelEncoder
from CNN import ProteinFeatureCNN
from CTF import ProEncoder
import torch


class FeatureFC:
    def __init__(self, WINDOW_P_UPLIMIT=3, CODING_FREQUENCY=True, VECTOR_REPETITION_CNN=1,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None,
                 pro_coding_length=399,
                 vector_repeatition_cnn=1):
        self.encoder = ProEncoder(WINDOW_P_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                                  TRUNCATION_LEN, PERIOD_EXTENDED)
        self.model = ProteinFeatureCNN(
            pro_coding_length=pro_coding_length,
            vector_repeatition_cnn=vector_repeatition_cnn
        )

    def read_tsv_file(self, tsv_file_path):
        print("开始读取 TSV 文件...")
        sequences = {}
        labels = {}
        with open(tsv_file_path, 'r') as f:
            header = f.readline().strip().split('\t')
            sequence_index = header.index('sequence')
            label_index = header.index('label')
            for line in f:
                line = line.strip().split('\t')
                sequence_name = len(sequences) + 1
                sequence = line[sequence_index]
                label = line[label_index]
                sequences[sequence_name] = sequence
                labels[sequence_name] = label
        print("TSV 文件读取完成。")
        return sequences, labels

    def convert_sequences_to_cnn_input(self, tsv_file_path):
        print("开始将序列转换为 CNN 输入...")
        sequences, labels = self.read_tsv_file(tsv_file_path)
        cnn_input_vectors = {}
        total_sequences = len(sequences)
        for i, (sequence_name, sequence) in enumerate(sequences.items()):
            cnn_input_vectors[sequence_name] = self.encoder.encode_conjoint_cnn(sequence)
            if (i + 1) % 100 == 0 or (i + 1) == total_sequences:
                print(f"已转换 {i + 1}/{total_sequences} 条序列。")
        all_vectors_list = []
        all_labels = []
        for sequence_name, vector in cnn_input_vectors.items():
            all_vectors_list.append(vector)
            all_labels.append(labels[sequence_name])
        all_vectors_array = np.array(all_vectors_list)
        print("序列转换为 CNN 输入完成。")
        return all_vectors_array, all_labels

    def extract_features(self, input_array):
        print("开始提取特征...")
        input_tensor = torch.from_numpy(input_array).float().permute(0, 2, 1)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            features = self.model(input_tensor)
        print("特征提取完成。")
        return features

    def transform_features_and_labels(self, features, labels):
        print("开始转换特征和标签...")
        num_samples_to_add = 658- features.size(0)
        if num_samples_to_add > 0:
            padding_features = torch.zeros(num_samples_to_add, features.size(1)).to(features.device)
            features = torch.cat((features, padding_features), dim=0)
            padding_labels = torch.zeros(num_samples_to_add, dtype=labels.dtype).to(labels.device)
            labels = torch.cat((labels, padding_labels), dim=0)
        elif num_samples_to_add < 0:
            features = features[:658, :]
            labels = labels[:658]
        features = features.unsqueeze(1).repeat(1, 98, 1)
        print("特征和标签转换完成。")
        return features, labels

    def save_features_to_npy(self, features, filename='../data/AT/polymerase/FC_AT.npy'):
        print(f"开始保存特征到 {filename}...")
        features_np = features.detach().cpu().numpy()
        np.save(filename, features_np)
        print(f"特征保存完成。")

    def save_labels_to_npy(self, labels, filename='../data/AT/polymerase/labelAT.npy'):
        print(f"开始保存标签到 {filename}...")
        labels_np = labels.detach().cpu().numpy()
        np.save(filename, labels_np)
        print(f"标签保存完成。")

    def process(self, tsv_file_path):
        input_array, all_labels = self.convert_sequences_to_cnn_input(tsv_file_path)
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(all_labels)
        labels = torch.from_numpy(encoded_labels).long()
        if torch.cuda.is_available():
            labels = labels.cuda()
        features = self.extract_features(input_array)
        features_match, labels_match = self.transform_features_and_labels(features, labels)
        self.save_features_to_npy(features_match)
        self.save_labels_to_npy(labels_match)
        print("扩展后的特征形状:", features_match.shape)
        print("扩展后的标签形状:", labels_match.shape)
        return features_match, labels_match


if __name__ == "__main__":
    feature_extractor = FeatureFC()
    tsv_file_path = "../data/AT/polymerase/AT_pro.tsv"
    feature_extractor.process(tsv_file_path)
