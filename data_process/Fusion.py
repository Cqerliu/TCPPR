import torch
import torch.nn as nn
import numpy as np


class FeatureFusion:
    def __init__(self, ft_path, fc_path, label_path1, label_path2,
                 d_model=64, num_heads=4, d_ff=128, dropout=0.1, batch_size=32):
        """
        初始化 FeatureFusion 类
        :param ft_path: FT 特征.npy 文件的路径
        :param fc_path: FC 特征.npy 文件的路径
        :param label_path1: 第一个标签.npy 文件的路径
        :param label_path2: 第二个标签.npy 文件的路径
        :param d_model: 模型维度
        :param num_heads: 多头注意力头数
        :param d_ff: 前馈网络维度
        :param dropout: 丢弃率
        :param batch_size: 批次大小
        """
        self.ft_path = ft_path
        self.fc_path = fc_path
        self.label_path1 = label_path1
        self.label_path2 = label_path2
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.batch_size = batch_size
        self.fused_vector = None
        self.fused_label = None

        self.model = nn.Sequential(
            self._create_feature_fusion(),
            nn.Flatten()
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _create_feature_fusion(self):
        """
        创建特征融合模型
        :return: FeatureFusionModule 模型实例
        """
        return FeatureFusionModule(self.d_model, self.num_heads, self.d_ff, self.dropout)

    def load_data(self):
        """
        加载 FT、FC 特征数据和标签数据
        """
        self.ft_tensor = self._load_npy_as_tensor(self.ft_path)
        self.fc_tensor = self._load_npy_as_tensor(self.fc_path)

        label_tensor1 = np.load(self.label_path1)
        label_tensor2 = np.load(self.label_path2)
        self.label_tensor = np.concatenate([label_tensor1, label_tensor2])
        self.label_tensor = torch.tensor(self.label_tensor).long()

        print(f"ft_tensor shape: {self.ft_tensor.shape}")
        print(f"fc_tensor shape: {self.fc_tensor.shape}")

    def _load_npy_as_tensor(self, file_path):
        """
        从.npy 文件加载数据并转换为 PyTorch 张量
        :param file_path:.npy 文件的路径
        :return: PyTorch 张量
        """
        array = np.load(file_path)
        tensor = torch.from_numpy(array).float()
        return tensor

    def fuse_features(self):
        """
        执行特征融合
        """
        fused_vectors = []
        fused_labels = []

        num_batches_ft = (self.ft_tensor.shape[0] + self.batch_size - 1) // self.batch_size
        num_batches_fc = (self.fc_tensor.shape[0] + self.batch_size - 1) // self.batch_size
        num_batches = max(num_batches_ft, num_batches_fc)

        for i in range(num_batches):
            start_idx_ft = i * self.batch_size
            end_idx_ft = min((i + 1) * self.batch_size, self.ft_tensor.shape[0])
            start_idx_fc = i * self.batch_size
            end_idx_fc = min((i + 1) * self.batch_size, self.fc_tensor.shape[0])

            ft_batch = self.ft_tensor[start_idx_ft:end_idx_ft]
            fc_batch = self.fc_tensor[start_idx_fc:end_idx_fc]
            label_batch = self.label_tensor[start_idx_ft:end_idx_ft]

            if torch.cuda.is_available():
                ft_batch = ft_batch.cuda()
                fc_batch = fc_batch.cuda()
                label_batch = label_batch.cuda()

            # 直接调用 FeatureFusionModule 的 forward 方法
            fusion_module = self.model[0]
            fused_batch = fusion_module(ft_batch, fc_batch)
            # 手动进行 Flatten 操作
            flattened_batch = torch.flatten(fused_batch, start_dim=1)

            fused_vectors.append(flattened_batch.cpu().detach())
            fused_labels.append(label_batch.cpu().detach())

            print(f"Processed batch {i + 1}/{num_batches}")

        self.fused_vector = torch.cat(fused_vectors, dim=0)
        self.fused_label = torch.cat(fused_labels, dim=0)

        print(f"Fused vector shape: {self.fused_vector.shape}")
        print(f"Fused label shape: {self.fused_label.shape}")

    def save_results(self, fused_vector_save_path='../fused/fused_AT.npy',
                     fused_label_save_path='../fused/fusedlb_AT.npy'):
        """
        保存融合后的特征向量和标签
        :param fused_vector_save_path: 融合后的特征向量保存路径
        :param fused_label_save_path: 融合后的标签保存路径
        """
        fused_vector_np = self.fused_vector.numpy()
        np.save(fused_vector_save_path, fused_vector_np)
        print(f"融合并展平后的特征向量已保存到 {fused_vector_save_path}")

        fused_label_np = self.fused_label.numpy()
        np.save(fused_label_save_path, fused_label_np)
        print(f"融合后的标签已保存到 {fused_label_save_path}")


class FeatureFusionModule(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(FeatureFusionModule, self).__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.self_attn3 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ft, fc):
        # 第一次多头注意力与融合
        attn_output1, _ = self.self_attn1(ft, ft, ft)
        attn_output1 = self.dropout(attn_output1)
        fused_features1 = self.layer_norm1(attn_output1 + ft)

        # 第二次多头注意力与融合
        attn_output2, _ = self.self_attn2(fused_features1, fc, fc)
        attn_output2 = self.dropout(attn_output2)
        fused_features2 = self.layer_norm2(attn_output2 + fused_features1 + fc)

        # 第三次多头注意力与融合
        attn_output3, _ = self.self_attn3(fused_features2, fc, fc)
        attn_output3 = self.dropout(attn_output3)
        fused_features3 = self.layer_norm3(attn_output3 + fused_features2 + fc)

        # 前馈网络处理
        ff_output = self.feed_forward(fused_features3)
        ff_output = self.dropout(ff_output)
        final_output = self.layer_norm4(ff_output + fused_features3)

        return final_output


if __name__ == "__main__":
    ft_path = '../data/AT/promoter/FT_AT.npy'
    fc_path = '../data/AT/polymerase/FC_AT.npy'
    label_path1 = '../data/AT/promoter/AT_labels.npy'
    label_path2 = '../data/AT/polymerase/labelAT.npy'

    feature_fusion = FeatureFusion(ft_path, fc_path, label_path1, label_path2)
    feature_fusion.load_data()
    feature_fusion.fuse_features()
    feature_fusion.save_results()
