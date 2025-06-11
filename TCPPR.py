import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


class TCPPR:
    def __init__(self):
        """
        初始化函数
        """
        self.input_dim = None
        self.fused_vector = None
        self.encoded_labels = None
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.fold_accuracies = []
        self.fold_precisions = []
        self.fold_recalls = []
        self.fold_f1s = []
        self.fold_aucs = []
        self.all_fprs = []
        self.all_tprs = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 添加设备属性
        self.output_features = []
        self.output_labels = []
        self.dataset_names = ["Escherichia coli K-12", "H. TATA", "H. Non_TATA", "Plants", "M. TATA", "M. Non_TATA"]

    def load_data(self, fused_vector_path, encoded_labels_file):
        """
        加载融合向量和标签数据的方法
        :param fused_vector_path: 融合向量.npy文件路径
        :param encoded_labels_file: 编码标签.npy文件路径
        """
        self.fused_vector = self.load_npy_as_tensor(fused_vector_path).to(self.device)  # 加载后直接移动到GPU
        print(f"Fused vector shape: {self.fused_vector.shape}")
        encoded_labels = np.load(encoded_labels_file)
        self.encoded_labels = torch.tensor(encoded_labels).long().to(self.device)  # 转换为tensor后移动到GPU
        assert len(self.encoded_labels) == self.fused_vector.size(0), "标签数量和融合向量数量不一致"
        self.input_dim = self.fused_vector.size(1)

    def load_npy_as_tensor(self, file_path):
        """
        从.npy 文件加载数据并转换为 PyTorch 张量
        :param file_path:.npy 文件的路径
        :return: PyTorch 张量
        """
        array = np.load(file_path)
        tensor = torch.from_numpy(array).float()
        return tensor

    def BinaryClassifier(self, input_dim):
        """
        构建二分类模型（多层感知机）
        :param input_dim: 输入维度
        :return: 二分类模型实例
        """

        class _BinaryClassifier(nn.Module):
            def __init__(self, input_dim):
                super(_BinaryClassifier, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)  # 增加神经元数量
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(128, 64)  # 增加一个隐藏层
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(64, 2)  # 二分类，输出维度为 2

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu2(out)
                out = self.fc3(out)
                return out

        classifier = _BinaryClassifier(input_dim).to(self.device)  # 初始化模型后移动到GPU
        return classifier

    def train_model(self, classifier, X_train, y_train, num_epochs=1000):
        """
        训练模型
        :param classifier: 模型实例
        :param X_train: 训练数据
        :param y_train: 训练标签
        :param num_epochs: 训练轮数，默认1000
        :return: 最佳模型状态和最小损失
        """
        criterion = nn.CrossEntropyLoss().to(self.device)  # 损失函数移动到GPU
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
        min_loss = float('inf')
        best_model_state = None
        for epoch in range(num_epochs):
            classifier.train()
            outputs = classifier(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新学习率
            scheduler.step()

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}, Current LR: {scheduler.get_last_lr()[0]:.6f}')

            # 检查是否为最小损失
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_model_state = classifier.state_dict()

        return best_model_state, min_loss

    def evaluate_model(self, classifier, X_test, y_test):
        """
        评估模型
        :param classifier: 模型实例
        :param X_test: 测试数据
        :param y_test: 测试标签
        :return: 准确率、精确率、召回率、F1值、假正率、真正率、AUC值、敏感性、特异性
        """
        classifier.eval()
        with torch.no_grad():
            test_outputs = classifier(X_test)
            _, test_predicted = torch.max(test_outputs, 1)
            test_predicted = test_predicted.cpu().numpy()
            y_test = y_test.cpu().numpy()

            # 保存输出层特征和标签
            self.output_features.extend(test_outputs.cpu().numpy())
            self.output_labels.extend(y_test)

            # 计算准确率
            test_accuracy = (test_predicted == y_test).sum() / len(y_test)

            # 计算预测概率
            test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

            # 计算混淆矩阵的指标
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, test_predicted)
            recall = recall_score(y_test, test_predicted)
            f1 = f1_score(y_test, test_predicted)

            # 计算 AUC
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, test_probs)
            roc_auc = auc(fpr, tpr)

            # 计算敏感性和特异性
            true_positives = np.sum((test_predicted == 1) & (y_test == 1))
            false_negatives = np.sum((test_predicted == 0) & (y_test == 1))
            false_positives = np.sum((test_predicted == 1) & (y_test == 0))
            true_negatives = np.sum((test_predicted == 0) & (y_test == 0))

            specificity = true_negatives / (true_negatives + false_positives) if (
                                                                                         true_negatives + false_positives) > 0 else 0

            return test_accuracy, precision, recall, f1, fpr, tpr, roc_auc, specificity

    def cross_validate(self):
        """
        执行五折交叉验证
        :return: 平均准确率、平均精确率、平均召回率、平均F1值、平均AUC值、平均敏感性、平均特异性
        """
        fold_sensitivities = []
        fold_specificities = []
        for fold, (train_index, test_index) in enumerate(self.kf.split(self.fused_vector)):
            print(f"Fold {fold + 1}/{self.kf.get_n_splits()}")
            X_train, X_test = self.fused_vector[train_index], self.fused_vector[test_index]
            y_train, y_test = self.encoded_labels[train_index], self.encoded_labels[test_index]

            classifier = self.BinaryClassifier(self.input_dim)
            best_model_state, _ = self.train_model(classifier, X_train, y_train)
            classifier.load_state_dict(best_model_state)

            test_accuracy, precision, recall, f1, fpr, tpr, roc_auc, specificity = self.evaluate_model(
                classifier, X_test, y_test)
            self.fold_accuracies.append(test_accuracy)
            self.fold_precisions.append(precision)
            self.fold_recalls.append(recall)
            self.fold_f1s.append(f1)
            self.fold_aucs.append(roc_auc)
            self.all_fprs.append(fpr)
            self.all_tprs.append(tpr)
            fold_specificities.append(specificity)

            print(f'Fold {fold + 1} Test Accuracy: {test_accuracy * 100:.2f}%')
            print(f'Fold {fold + 1} Precision: {precision * 100:.2f}%')
            print(f'Fold {fold + 1} Recall: {recall * 100:.2f}%')
            print(f'Fold {fold + 1} F1-score: {f1 * 100:.2f}%')
            print(f'Fold {fold + 1} AUC: {roc_auc:.4f}')
            print(f'Fold {fold + 1} Specificity: {specificity * 100:.2f}%')

        average_accuracy = np.mean(self.fold_accuracies)
        average_precision = np.mean(self.fold_precisions)
        average_recall = np.mean(self.fold_recalls)
        average_f1 = np.mean(self.fold_f1s)
        average_auc = np.mean(self.fold_aucs)
        average_specificity = np.mean(fold_specificities)

        print(f'Average Test Accuracy across all folds: {average_accuracy * 100:.2f}%')
        print(f'Average Precision across all folds: {average_precision * 100:.2f}%')
        print(f'Average Recall across all folds: {average_recall * 100:.2f}%')
        print(f'Average F1-score across all folds: {average_f1 * 100:.2f}%')
        print(f'Average AUC across all folds: {average_auc:.4f}')
        print(f'Average  Specificity across all folds: {average_specificity * 100:.2f}%')

        return average_accuracy, average_precision, average_recall, average_f1, average_auc, average_specificity

    def plot_roc(self):
        """
        绘制AUC曲线
        """
        plt.figure()
        for i in range(len(self.all_fprs)):
            plt.plot(self.all_fprs[i], self.all_tprs[i], label=f'Fold {i + 1} (AUC = {self.fold_aucs[i]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - All Folds')
        plt.legend(loc="lower right")
        plt.show()




if __name__ == "__main__":
    # 融合向量和标签文件的路径
    fused_vector_path = 'fused_features/fused_AT.npy'
    encoded_labels_file = 'fused_features/fusedlb_AT.npy'

    # 创建 TCPPR 类的实例
    crptcnn = TCPPR()

    # 加载数据
    crptcnn.load_data(fused_vector_path, encoded_labels_file)

    # 执行五折交叉验证并获取平均评估指标
    average_accuracy, average_precision, average_recall, average_f1, average_auc, average_specificity = crptcnn.cross_validate()

    # 绘制 ROC 曲线
    crptcnn.plot_roc()

