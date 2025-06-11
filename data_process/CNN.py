import torch
import torch.nn as nn

class ProteinFeatureCNN(nn.Module):
    def __init__(self, pro_coding_length, vector_repeatition_cnn):
        super(ProteinFeatureCNN, self).__init__()
        if isinstance(vector_repeatition_cnn, int):
            vec_len_p = vector_repeatition_cnn
        else:
            vec_len_p = vector_repeatition_cnn[0]

        self.conv1 = nn.Conv1d(in_channels=vec_len_p, out_channels=64, kernel_size=7, stride=1)  # 增加卷积核数量和大小
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1)  # 增加卷积核数量
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=96, kernel_size=5, stride=1)  # 调整卷积核数量和大小
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.bn3 = nn.BatchNorm1d(96)
        self.drop3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv1d(in_channels=96, out_channels=64, kernel_size=4, stride=1)  # 增加一个卷积层
        self.relu4 = nn.LeakyReLU(negative_slope=0.01)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_conv_output((vec_len_p, pro_coding_length)), 128)  # 增加全连接层神经元数量
        self.drop5 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)  # 增加一个全连接层
        self.drop6 = nn.Dropout(0.2)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.relu3(self.conv3(x))
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.relu4(self.conv4(x))
        x = self.bn4(x)
        x = self.drop4(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop5(x)
        x = self.fc2(x)
        x = self.drop6(x)
        return x