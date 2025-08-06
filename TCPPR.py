import random
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from encode import PromoterEncoder, ProEncoder
from TCPPR_module import TransformerFeature, MLP, CNNFeature, FeatureFusion
import numpy as np
import os
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, x, fc, y):
        self.x = x
        self.fc = fc
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_tensor = torch.from_numpy(self.x[idx]).float()
        fc_tensor = torch.from_numpy(self.fc[idx]).float()
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, fc_tensor, y_tensor


def main():
    # set seed
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存模型的目录
    os.makedirs('best_model', exist_ok=True)

    # read original promoter datas
    promoter_data = pd.read_csv('data/coli/promoter/coli.tsv', sep='\t')
    promoter_sequences = promoter_data['sequence'].values
    y = promoter_data['label'].values.astype(np.int64)
    total_samples = len(y)
    print(f'原始启动子样本数: {total_samples}')
    # read original RNAP datas
    rnap_data = pd.read_csv('data/coli/polymerase/coli_pro.tsv', sep='\t')
    rnap_sequences = rnap_data['sequence'].values
    print(f'原始RNAP样本数: {len(rnap_sequences)}')
    # verify sample consistency
    assert len(promoter_sequences) == len(y) == len(rnap_sequences), "样本数不匹配"

    # synchronization randomly disrupts all raw data
    n_samples = len(y)
    shuffled_indices = np.random.permutation(n_samples)
    promoter_sequences = promoter_sequences[shuffled_indices]
    rnap_sequences = rnap_sequences[shuffled_indices]
    y = y[shuffled_indices]
    # verify sample consistency
    assert len(promoter_sequences) == len(y) == len(rnap_sequences), "打乱后数据长度不匹配"
    print("数据已同步打乱，启动子与RNAP对应关系保持一致")

    # five-fold cross-validation
    kf = KFold(n_splits=5, shuffle=False)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [], 'mse': []}
    PRINT_SAMPLES = 10
    def clear_memory():
        """clear memory"""
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    for fold, (train_index, test_index) in enumerate(kf.split(promoter_sequences)):
        print(f"\n===== 第 {fold + 1} 折 =====")
        # divide data sets
        train_promoter = promoter_sequences[train_index]
        test_promoter = promoter_sequences[test_index]
        train_rnap = rnap_sequences[train_index]
        test_rnap = rnap_sequences[test_index]
        train_y = y[train_index]
        test_y = y[test_index]

        # encode promoter datas
        print("编码启动子数据...")
        encoder = PromoterEncoder(kmer=3, vector_size=100)
        x_train, _ = encoder.run_pipeline(sequences=train_promoter, labels=train_y)
        x_train = x_train.astype(np.float32)
        # the test set is encoded using the trained encoder
        trained_word_vectors = encoder.word_vectors
        trained_sequence_length = encoder.sequence_length
        trained_label_encoder = encoder.label_encoder

        test_encoder = PromoterEncoder(kmer=3, vector_size=100)
        test_encoder.word_vectors = trained_word_vectors
        test_encoder.sequence_length = trained_sequence_length
        test_encoder.label_encoder = trained_label_encoder
        test_encoder.set_sequences_labels(sequences=test_promoter, labels=test_y)
        test_encoder.generate_kmers()
        x_test, _ = test_encoder.get_transformer_input()
        x_test = x_test.astype(np.float32)
        print(f'启动子训练集编码大小: {x_train.shape}')
        print(f'启动子测试集编码大小: {x_test.shape}')
        # encode RNAP datas
        print("编码RNAP数据...")
        fc_encode = ProEncoder(VECTOR_REPETITION_CNN=x_train.shape[1])
        fc_train = np.array([
            fc_encode.encode_conjoint_cnn(seq).squeeze(0).cpu().numpy().astype(np.float32)
            for seq in train_rnap], dtype=np.float32)
        fc_test = np.array([
            fc_encode.encode_conjoint_cnn(seq).squeeze(0).cpu().numpy().astype(np.float32)
            for seq in test_rnap], dtype=np.float32)
        print(f'RNAP训练集编码大小: {fc_train.shape}')
        print(f'RNAP测试集编码大小: {fc_test.shape}')

        # clean up coded variables
        del encoder, test_encoder, fc_encode
        clear_memory()
        # creat datasets and data loaders
        train_dataset = CustomDataset(x_train, fc_train, train_y)
        test_dataset = CustomDataset(x_test, fc_test, test_y)
        batch_size = 32
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=0)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=0)

        # model initialization
        transformer = TransformerFeature(input_dim=x_train.shape[2],max_seq_len=x_train.shape[1],
            depth=4, heads=5, dim_head=16,attn_dropout=0.1, ff_dropout=0.1, use_projection=True).to(device)
        cnn = CNNFeature(input_length=fc_train.shape[2],input_channels=fc_train.shape[1],
            feature_dim=x_train.shape[2],conv_filters=[256, 128, 64],conv_kernels=[7, 7, 5],pool_sizes=[2, 2, 2],dropout_rate=0.2).to(device)
        fusion = FeatureFusion(feature_dim=x_train.shape[2],dropout=0.2).to(device)
        mlp = MLP(input_dim=x_train.shape[2],num_classes=2).to(device)

        # define the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        weight_decay = 0.0001
        optimizer = torch.optim.Adam(list(transformer.parameters()) + list(cnn.parameters()) +list(fusion.parameters()) + list(mlp.parameters()),lr=0.00001, weight_decay=weight_decay)

        # train the model
        num_epochs = 60
        best_accuracy = 0.0
        best_model_params = {'transformer': None, 'cnn': None, 'fusion': None, 'mlp': None}

        for epoch in range(num_epochs):
            transformer.train()
            cnn.train()
            fusion.train()
            mlp.train()
            running_loss = 0.0
            all_train_predicted = []
            all_train_labels = []
            print_train_labels = (epoch == 0)

            for batch_idx, (batch_X, batch_fc, batch_y) in enumerate(train_dataloader):
                batch_X = batch_X.to(device, non_blocking=True)
                batch_fc = batch_fc.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad()

                # forward propagation
                promoter_features = transformer(batch_X)
                rnap_features = cnn(batch_fc)
                fused_features = fusion(promoter_features, rnap_features)
                logits = mlp(fused_features)

                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # record training results
                predicted = torch.argmax(logits, dim=1)
                all_train_predicted.extend(predicted.cpu().numpy())
                all_train_labels.extend(batch_y.cpu().numpy())

                if print_train_labels and batch_idx == 0:
                    print(f"\n训练集 - 第1个批次前{PRINT_SAMPLES}个样本:")
                    print(f"原始标签: {batch_y[:PRINT_SAMPLES].cpu().numpy()}")
                    print(f"预测标签: {predicted[:PRINT_SAMPLES].cpu().numpy()}\n")
                    print_train_labels = False

            epoch_loss = running_loss / len(train_dataloader)
            train_accuracy = accuracy_score(all_train_labels, all_train_predicted)
            print(f"Epoch {epoch + 1}/{num_epochs}, train_loss: {epoch_loss:.3f}, train_acc: {train_accuracy:.3f}")
            clear_memory()

            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy
                best_model_params['transformer'] = transformer.state_dict()
                best_model_params['cnn'] = cnn.state_dict()
                best_model_params['fusion'] = fusion.state_dict()
                best_model_params['mlp'] = mlp.state_dict()
                print(f"  已更新最佳模型 (acc: {best_accuracy:.3f})")

        # test the model
        transformer.load_state_dict(best_model_params['transformer'])
        cnn.load_state_dict(best_model_params['cnn'])
        fusion.load_state_dict(best_model_params['fusion'])
        mlp.load_state_dict(best_model_params['mlp'])

        transformer.eval()
        cnn.eval()
        fusion.eval()
        mlp.eval()
        all_predicted = []
        all_labels = []
        all_probs = []
        print_test_labels = True

        with torch.no_grad():
            for batch_idx, (batch_X, batch_fc, batch_y) in enumerate(test_dataloader):
                batch_X = batch_X.to(device, non_blocking=True)
                batch_fc = batch_fc.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                promoter_features = transformer(batch_X)
                rnap_features = cnn(batch_fc)
                fused_features = fusion(promoter_features, rnap_features)
                logits = mlp(fused_features)

                predicted = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]

                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                if print_test_labels and batch_idx == 0:
                    print(f"\n测试集 - 第1个批次前{PRINT_SAMPLES}个样本:")
                    print(f"原始标签: {batch_y[:PRINT_SAMPLES].cpu().numpy()}")
                    print(f"预测标签: {predicted[:PRINT_SAMPLES].cpu().numpy()}\n")
                    print_test_labels = False

        # calculation of assessment indicators
        accuracy = accuracy_score(all_labels, all_predicted)
        precision = precision_score(all_labels, all_predicted)
        recall = recall_score(all_labels, all_predicted)
        f1 = f1_score(all_labels, all_predicted)
        auc = roc_auc_score(all_labels, all_probs)
        mse = mean_squared_error(all_labels, all_predicted)

        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)
        metrics['mse'].append(mse)

        print(f"\n第 {fold + 1} 折测试结果:")
        print(f"  准确率 (Accuracy): {accuracy:.3f}")
        print(f"  精确率 (Precision): {precision:.3f}")
        print(f"  召回率 (Recall): {recall:.3f}")
        print(f"  F1分数 (F1 Score): {f1:.3f}")
        print(f"  AUC: {auc:.3f}")

        # clean up variables
        del x_train, x_test, fc_train, fc_test, train_dataset, test_dataset
        del transformer, cnn, fusion, mlp, optimizer
        clear_memory()

    # print final result
    print("\n===== 五折交叉验证最终结果汇总 =====")
    print(f"平均准确率 (Accuracy): {np.mean(metrics['accuracy']):.3f} ± {np.std(metrics['accuracy']):.3f}")
    print(f"平均精确率 (Precision): {np.mean(metrics['precision']):.3f} ± {np.std(metrics['precision']):.3f}")
    print(f"平均召回率 (Recall): {np.mean(metrics['recall']):.3f} ± {np.std(metrics['recall']):.3f}")
    print(f"平均F1分数 (F1 Score): {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
    print(f"平均AUC: {np.mean(metrics['auc']):.3f} ± {np.std(metrics['auc']):.3f}")
    print(f"平均均方误差 (MSE): {np.mean(metrics['mse']):.3f} ± {np.std(metrics['mse']):.3f}")


if __name__ == '__main__':
    main()
