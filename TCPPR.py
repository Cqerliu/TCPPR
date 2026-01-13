import random
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
import seaborn as sns
from sklearn.metrics import confusion_matrix
from encode import PromoterEncoder, ProEncoder
import pandas as pd
import warnings
from collections import defaultdict
from downTCPPR_module import TransformerFeature, MLP, CNNFeature, FeatureFusion
from data_split import create_promoter_subsets_with_replacement
from visualization import *
import torch
import  torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class FocalLoss(nn.Module):
    """place greater emphasis on hard samples"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class PrototypeLoss(nn.Module):
    """ align RNAP and promoter representations in the feature space"""
    def __init__(self, lambda_align=0.1):
        super().__init__()
        self.focal = FocalLoss(gamma=2)
        self.align_loss = nn.CosineEmbeddingLoss(margin=0.0)
        self.lambda_align = lambda_align

    def forward(self, logits, dna_feat, rnap_feat, targets):
        # 1. main classification loss
        cls_loss = self.focal(logits, targets)
        # 2. Alignment Loss
        # 1 indicates that the two feature vectors are similar and should be pulled closer in the embedding space.
        align_targets = targets.float() * 2 - 1
        # ensure dimensional alignment
        if dna_feat.dim() > 2: dna_feat = dna_feat.max(dim=1)[0]
        if rnap_feat.dim() > 2: rnap_feat = rnap_feat.max(dim=1)[0]

        feat_loss = self.align_loss(dna_feat, rnap_feat, align_targets)
        total_loss = cls_loss + self.lambda_align * feat_loss
        return total_loss, cls_loss, feat_loss

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


def clear_memory(model_list=None):
    if model_list is not None:
        for model in model_list:
            if model is not None:
                del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("Memory cleanup complete")


def deduplicate_samples(promoter_seqs, rnap_seqs, labels, predictions, probabilities=None):
    input_lengths = [len(promoter_seqs), len(rnap_seqs), len(labels), len(predictions)]
    if probabilities is not None:
        input_lengths.append(len(probabilities))
    if len(set(input_lengths)) != 1:
        raise ValueError(f"Input list lengths are inconsistent! Each length：{input_lengths}")

    valid_promoter_indices = []
    for idx, prom_seq in enumerate(promoter_seqs):
        if prom_seq is not None and str(prom_seq).strip() != "":
            valid_promoter_indices.append(idx)

    # Promoter de-duplication based on validated samples
    seen_promoters = set()
    unique_indices = []
    for i in valid_promoter_indices:
        prom_seq = promoter_seqs[i]
        if prom_seq not in seen_promoters:
            seen_promoters.add(prom_seq)
            unique_indices.append(i)

    dedup_promoter = [promoter_seqs[i] for i in unique_indices]
    dedup_rnap = [rnap_seqs[i] for i in unique_indices]
    dedup_labels = [labels[i] for i in unique_indices]
    dedup_preds = [predictions[i] for i in unique_indices]
    dedup_probs = [probabilities[i] for i in unique_indices] if probabilities is not None else None

    stats = {'original_total_count': len(promoter_seqs),
        'valid_promoter_count': len(valid_promoter_indices),  # Number of samples with no promoter abnormality
        'unique_promoter_count': len(unique_indices),  # Final sample size after de-weighting
        'deleted_duplicate_count': len(valid_promoter_indices) - len(unique_indices),  # Number of true replicates
        'deleted_invalid_count': len(promoter_seqs) - len(valid_promoter_indices)  # Number of samples filtered for promoter anomalies
    }

    return {
        'promoter': dedup_promoter,
        'rnap': dedup_rnap,
        'labels': dedup_labels,
        'predictions': dedup_preds,
        'probabilities': dedup_probs,
        'original_count': stats['original_total_count'],
        'unique_count': stats['unique_promoter_count'],
        'stats_detail': stats
    }

def plot_overall_confusion_matrix(dedup_data, avg_metrics, species_name, save_dir):
    cm = confusion_matrix(dedup_data['labels'], dedup_data['predictions'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='RdBu_r',  
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['True 0', 'True 1'],
        annot_kws={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'},  
        cbar=True,  
        cbar_kws={'shrink': 0.8, 'aspect': 20, 'pad': 0.02}, 
        linewidths=2,  
        linecolor='white', 
        square=True
    )
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title(f'{species_name} - Overall Confusion Matrix (Unique Samples: {dedup_data["unique_count"]})',
              fontsize=16, fontweight='bold', pad=20)  

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    save_path = os.path.join(save_dir, f'{species_name}_overall_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The overall confusion matrix after de-weighting has been saved to the: {save_path}")

def process_subset(subset_idx, promoter_subset, rnap_subset, device, base_seed=41):
    print(f"\n===== Processing a subset of promoters {subset_idx + 1} =====")
    promoter_sequences = promoter_subset['sequence'].values
    y = promoter_subset['label'].values.astype(np.int64)
    subset_size = len(y)
    print(f"Current subset size: {subset_size}")
    rnap_sequences = rnap_subset['sequence'].values

    shuffled_indices = np.random.permutation(subset_size)
    promoter_sequences = promoter_sequences[shuffled_indices]
    rnap_sequences = rnap_sequences[shuffled_indices]
    y = y[shuffled_indices]
    print("The data has been synchronized and disrupted")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []} 
    PRINT_SAMPLES = 10

    fold_test_data = {
        'promoter': [], 'rnap': [], 'labels': [], 'predictions': [], 'probabilities': []  
    }

    best_model = None
    global_x_test = None
    global_fc_test = None
    last_test_promoter = None
    last_test_rnap = None
    all_fold_test_labels = []

    prom_original_len = len(promoter_subset['sequence'].iloc[0]) if not promoter_subset.empty else 0
    rnap_original_len = len(rnap_subset['sequence'].iloc[0]) if not rnap_subset.empty else 0
    print(f"Original promoter length: {prom_original_len}, Original RNAP length: {rnap_original_len}")

    for fold, (train_index, test_index) in enumerate(kf.split(promoter_sequences, y)):
        print(f"\n----- Fold {fold + 1}  -----")
        train_promoter = promoter_sequences[train_index]
        test_promoter = promoter_sequences[test_index]
        train_rnap = rnap_sequences[train_index]
        test_rnap = rnap_sequences[test_index]
        train_y = y[train_index]
        test_y = y[test_index]
        all_fold_test_labels.extend(test_y)

        print("Encoded promoter data...")
        encoder = PromoterEncoder(kmer=5, vector_size=100)
        x_train, _ = encoder.run_pipeline(sequences=train_promoter, labels=train_y)
        x_train = x_train.astype(np.float32)
        test_encoder = PromoterEncoder(kmer=5, vector_size=100)
        test_encoder.word_vectors = encoder.word_vectors
        test_encoder.sequence_length = encoder.sequence_length
        test_encoder.label_encoder = encoder.label_encoder
        test_encoder.set_sequences_labels(sequences=test_promoter, labels=test_y)
        test_encoder.generate_kmers()
        x_test, _ = test_encoder.get_transformer_input()
        x_test = x_test.astype(np.float32)

        print("Encoded RNAP data...")
        fc_encode = ProEncoder(VECTOR_REPETITION_CNN=x_train.shape[1])
        fc_train = np.array([fc_encode.encode_conjoint_cnn(seq).squeeze(0).cpu().numpy().astype(np.float32)
                             for seq in train_rnap], dtype=np.float32)
        fc_test = np.array([fc_encode.encode_conjoint_cnn(seq).squeeze(0).cpu().numpy().astype(np.float32)
                            for seq in test_rnap], dtype=np.float32)

        del test_encoder, fc_encode
        clear_memory()

        train_dataset = CustomDataset(x_train, fc_train, train_y)
        test_dataset = CustomDataset(x_test, fc_test, test_y)
        batch_size = 64
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

        transformer = TransformerFeature(input_dim=x_train.shape[2], max_seq_len=x_train.shape[1],
                                         depth=4, heads=5, dim_head=16, attn_dropout=0.1, ff_dropout=0.1,
                                         use_projection=True).to(device)
        cnn = CNNFeature(input_length=fc_train.shape[2], input_channels=fc_train.shape[1],
                         feature_dim=x_train.shape[2], conv_filters=[256, 128],
                         conv_kernels=[5, 7], pool_sizes=[2, 2], dropout_rate=0.2).to(device)
        fusion = FeatureFusion(feature_dim=x_train.shape[2],heads=4,dim_head=16,dropout=0.2).to(device)
        mlp = MLP(input_dim=x_train.shape[2], num_classes=2).to(device)
        mlp_dna_only = MLP(input_dim=x_train.shape[2], num_classes=2).to(device)
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = PrototypeLoss(lambda_align=0.1).to(device)  
        weight_decay = 0.0001
        optimizer = torch.optim.Adam(list(transformer.parameters()) + list(cnn.parameters()) +
                                     list(fusion.parameters()) + list(mlp.parameters()) + list(mlp_dna_only.parameters()),
                                     lr=1e-4, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        num_epochs = 60
        best_accuracy = 0.0
        current_best_model = None

        for epoch in range(num_epochs):
            transformer.train()
            cnn.train()
            fusion.train()
            mlp.train()
            mlp_dna_only.train()

            running_loss = 0.0
            all_train_predicted = []
            all_train_labels = []
            print_train_labels = (epoch == 0)

            for batch_idx, (batch_X, batch_fc, batch_y) in enumerate(train_dataloader):
                batch_X = batch_X.to(device, non_blocking=True)
                batch_fc = batch_fc.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad()
                promoter_features = transformer(batch_X)
                # promoter_features = promoter_features.max(dim=1)[0]
                rnap_features = cnn(batch_fc)
                r_feat_for_fusion = rnap_features
                if random.random() < 0.3:
                    r_feat_for_fusion = torch.zeros_like(rnap_features)
                fusion_features, attn_weights1, attn_weights2 = fusion(promoter_features, r_feat_for_fusion)

                logits = mlp(fusion_features)
                # loss = criterion(logits, batch_y)

                # A. Fusion classification loss
                loss_fusion, _, _ = criterion(logits, promoter_features, rnap_features, batch_y)
                # B. Loss of independent categorization of DNA branches
                p_feat_global = promoter_features.max(dim=1)[0]
                logits_dna = mlp_dna_only(p_feat_global)
                loss_dna = F.cross_entropy(logits_dna, batch_y)
                # C. total_loss
                total_loss = loss_fusion + 0.7 * loss_dna

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)

                optimizer.step()
                # running_loss += loss.item()
                running_loss += total_loss.item()

                predicted = torch.argmax(logits, dim=1)
                all_train_predicted.extend(predicted.cpu().numpy())
                all_train_labels.extend(batch_y.cpu().numpy())

                if print_train_labels and batch_idx == 0:
                    print(f"\ntraining set - The first {PRINT_SAMPLES}samples in the first batch:")
                    print(f"original label: {batch_y[:PRINT_SAMPLES].cpu().numpy()}")
                    print(f"Predictive labeling: {predicted[:PRINT_SAMPLES].cpu().numpy()}\n")

                scheduler.step()

            epoch_loss = running_loss / len(train_dataloader)

            train_accuracy = accuracy_score(all_train_labels, all_train_predicted)
            print(f"Epoch {epoch + 1}/{num_epochs}, train_loss: {epoch_loss:.3f}, train_acc: {train_accuracy:.3f}")

            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy
                current_best_model = {
                    'transformer': transformer.state_dict(),
                    'cnn': cnn.state_dict(),
                    'fusion': fusion.state_dict(),
                    'mlp': mlp.state_dict()
                }
                print(f"  Updated with the best models (acc: {best_accuracy:.3f})")

        transformer.load_state_dict(current_best_model['transformer'])
        cnn.load_state_dict(current_best_model['cnn'])
        fusion.load_state_dict(current_best_model['fusion'])
        mlp.load_state_dict(current_best_model['mlp'])

        transformer.eval()
        cnn.eval()
        fusion.eval()
        mlp.eval()
        fold_predicted = []
        fold_labels = []
        fold_probabilities = []  

        with torch.no_grad():
            for batch_X, batch_fc, batch_y in test_dataloader:
                batch_X = batch_X.to(device)
                batch_fc = batch_fc.to(device)
                batch_y = batch_y.to(device)

                promoter_features = transformer(batch_X)
                # promoter_features = promoter_features.max(dim=1)[0]
                rnap_features = cnn(batch_fc)
                fusion_features, _, _ = fusion(promoter_features, rnap_features)
                logits = mlp(fusion_features)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  
                predicted = torch.argmax(logits, dim=1)

                fold_predicted.extend(predicted.cpu().numpy())
                fold_labels.extend(batch_y.cpu().numpy())
                fold_probabilities.extend(probs)  

        fold_test_data['promoter'].extend(test_promoter)
        fold_test_data['rnap'].extend(test_rnap)
        fold_test_data['labels'].extend(fold_labels)
        fold_test_data['predictions'].extend(fold_predicted)
        fold_test_data['probabilities'].extend(fold_probabilities)  

        accuracy = accuracy_score(fold_labels, fold_predicted)
        precision = precision_score(fold_labels, fold_predicted, zero_division=0)
        recall = recall_score(fold_labels, fold_predicted, zero_division=0)
        f1 = f1_score(fold_labels, fold_predicted, zero_division=0)
        if len(np.unique(fold_labels)) == 2:
            auc = roc_auc_score(fold_labels, fold_probabilities)
        else:
            auc = 0.0  
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)  

        print(f"\nTest results of fold {fold + 1}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  AUC: {auc:.3f}")  # 显示AUC

        if fold == kf.n_splits - 1:
            best_model = {
                'transformer': transformer,
                'cnn': cnn,
                'fusion': fusion,
                'mlp': mlp,
                'encoder': encoder
            }
            global_x_test = x_test.copy()
            global_fc_test = fc_test.copy()
            last_test_promoter = test_promoter
            last_test_rnap = test_rnap

        model_list = [transformer, cnn, fusion, mlp]
        clear_memory(model_list)
        del x_train, fc_train, train_dataset, train_dataloader
        clear_memory()

    subset_dedup = deduplicate_samples(
        fold_test_data['promoter'], fold_test_data['rnap'],
        fold_test_data['labels'], fold_test_data['predictions'],
        fold_test_data['probabilities'])

    subset_output = { 'metrics': metrics,
        'rnap_test_data': torch.from_numpy(global_fc_test).float().to(device) if global_fc_test is not None else None,
        'test_labels': np.array(all_fold_test_labels),'rnap_test_seqs': last_test_rnap,
        'best_model': best_model,'fold_test_data': subset_dedup}

def main():
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    os.makedirs('best_model', exist_ok=True)
    os.makedirs('visualizations3', exist_ok=True)
    os.makedirs('visualizations3/withRNAP_HM', exist_ok=True)
    species_vis_dir = "visualizations3/withRNAP_HM"
    os.makedirs(species_vis_dir, exist_ok=True)

    try:
        promoter_data = pd.read_csv('data/HM/promoter/HM.tsv', sep='\t')
        rnap_data = pd.read_csv('data/HM/polymerase/HM_pron_243bp.tsv', sep='\t')
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    species_name = "H. sapiens"

    promoter_subsets = create_promoter_subsets_with_replacement(promoter_data, rnap_data,seed_val)
    num_subsets = len(promoter_subsets)
    print(f"The final number of subsets generated: {num_subsets}")

    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []} 
    all_subsets_top_kmer = defaultdict(list)
    all_subsets_rnap_data = []
    all_subsets_test_labels = []
    all_subsets_rnap_seqs = []
    final_best_model = None

    all_test_data = { 'promoter': [], 'rnap': [], 'labels': [], 'predictions': [], 'probabilities': [] }

    for i, subset in enumerate(promoter_subsets):
        subset_output = process_subset(i, subset, rnap_data, device, seed_val)
        subset_metrics = subset_output['metrics']
        subset_top_kmer = subset_output['top_kmer']
        subset_rnap_data = subset_output['rnap_test_data']
        subset_test_labels = subset_output['test_labels']
        subset_rnap_seqs = subset_output['rnap_test_seqs']
        subset_best_model = subset_output['best_model']
        subset_dedup = subset_output['fold_test_data']

        for key in all_test_data:
            all_test_data[key].extend(subset_dedup[key])

        for metric in all_metrics:
            all_metrics[metric].append(np.mean(subset_metrics[metric]))

        if subset_top_kmer is not None:
            for kmer, score in subset_top_kmer:
                all_subsets_top_kmer[kmer].append(score)

        if subset_rnap_data is not None:
            all_subsets_rnap_data.append(subset_rnap_data)
        if subset_test_labels is not None:
            all_subsets_test_labels.append(subset_test_labels)
        if subset_rnap_seqs is not None and len(subset_rnap_seqs) > 0:
            all_subsets_rnap_seqs.append(subset_rnap_seqs)

        if subset_best_model is not None:
            final_best_model = subset_best_model

        print(f"\nAverage results of 5-fold cross-validation for subset {i + 1}/{num_subsets} :")
        print(f"  Average Accuracy: {np.mean(subset_metrics['accuracy']):.3f}")
        print(f"  Average Precision: {np.mean(subset_metrics['precision']):.3f}")
        print(f"  Average Recall: {np.mean(subset_metrics['recall']):.3f}")
        print(f"  Average F1-score: {np.mean(subset_metrics['f1']):.3f}")
        print(f"  Average AUC: {np.mean(subset_metrics['auc']):.3f}") 

    avg_metrics = {
        'accuracy': np.mean(all_metrics['accuracy']),
        'precision': np.mean(all_metrics['precision']),
        'recall': np.mean(all_metrics['recall']),
        'f1': np.mean(all_metrics['f1']),
        'auc': np.mean(all_metrics['auc'])  
    }

    final_dedup = deduplicate_samples(
        all_test_data['promoter'], all_test_data['rnap'],
        all_test_data['labels'], all_test_data['predictions'],
        all_test_data['probabilities']
    )
    print(f"\nGlobal de-duplication complete: original sample {final_dedup['original_count']} → unique sample {final_dedup['unique_count']}")

    if len(np.unique(final_dedup['labels'])) == 2 and final_dedup['probabilities'] is not None:
        global_auc = roc_auc_score(final_dedup['labels'], final_dedup['probabilities'])
        print(f"AUC of samples after global de-weighting: {global_auc:.3f}")
    else:
        global_auc = 0.0

    # ---------------------- Species-level visualization ----------------------
    print(f"\n===== Begin species-level visualization and analysis（{species_name}）=====")

    # 1. Plotting the de-weighted confusion matrix based on the average results
    plot_overall_confusion_matrix(final_dedup, avg_metrics, species_name, species_vis_dir)

    # 2. Species-level SHAP analysis
    if len(all_subsets_rnap_data) > 0 and final_best_model is not None:
        plot_shap_rnap_analysis_species_level(
            model={'cnn': final_best_model['cnn'], 'mlp': final_best_model['mlp']},
            all_subset_data=all_subsets_rnap_data,
            all_subset_labels=all_subsets_test_labels,
            all_subset_rnap_seqs=all_subsets_rnap_seqs,
            device=device,
            species_name=species_name,
            save_dir=species_vis_dir
        )

    if final_best_model is not None:
        print(f"\n===== Start calculating species-level modal contributions（{species_name}）=====")

        try:
            # Step 1: Selection of the sample
            sample_num = min(100, len(final_dedup['promoter']))
            sample_promoters = final_dedup['promoter'][:sample_num]
            sample_rnaps = final_dedup['rnap'][:sample_num]
            sample_labels = final_dedup['labels'][:sample_num]
            print(f"{sample_num} samples selected for SHAP analysis")

            # Step 2: Code the Promoter
            encoder = final_best_model['encoder']
            encoder.set_sequences_labels(sequences=sample_promoters, labels=sample_labels)
            encoder.generate_kmers()
            x_vis, _ = encoder.get_transformer_input()
            x_vis = x_vis.astype(np.float32)

            #Step 3: Code the RNAP
            fc_encode_vis = ProEncoder(VECTOR_REPETITION_CNN=x_vis.shape[1])
            fc_vis_list = []
            for seq in sample_rnaps:
                encoded_tensor = fc_encode_vis.encode_conjoint_cnn(seq)
                if encoded_tensor.dim() > 2:  
                    encoded_tensor = encoded_tensor.squeeze(0)
                fc_vis_list.append(encoded_tensor.cpu().numpy())
            fc_vis = np.array(fc_vis_list, dtype=np.float32)

            print(f"Data readiness: Promoter {x_vis.shape}, RNAP {fc_vis.shape}")

            #Step 4: Call the multisample calculation function
            contribution_results = calculate_contribution_scores(
                model_dict=final_best_model,
                x_data=x_vis,
                fc_data=fc_vis,
                device=device,
                n_samples=5 
            )
            prom_score = contribution_results['avg_promoter']
            rnap_score = contribution_results['avg_rnap']

            # Step 5: Plotting Contribution Comparisons
            print(f"平均得分 - Promoter: {prom_score:.2f}, RNAP: {rnap_score:.2f}")
            plot_aesthetic_donut(
                prom_score=prom_score,
                rnap_score=rnap_score,
                species_name=species_name,
                save_dir=species_vis_dir
            )

            # Step 6: Validate the representativeness of the 100 samples
            sample_sizes = [10, 20, 40, 60, 100]
            subsample_results = []
            full_data_results = {
                'promoter': contribution_results['all_promoter'],
                'rnap': contribution_results['all_rnap']
            }

            for size in sample_sizes:
                indices = np.random.choice(len(x_vis), size=size, replace=False)
                x_sub = x_vis[indices]
                fc_sub = fc_vis[indices]

                sub_result = calculate_contribution_scores(
                    model_dict=final_best_model,
                    x_data=x_sub,fc_data=fc_sub,device=device, n_samples=5)
                subsample_results.append({
                    'promoter': sub_result['all_promoter'],'rnap': sub_result['all_rnap'] })

        except Exception as e:
            print(f"Error calculating SHAP contribution: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------- Summary of results ----------------------
    print("\n===== Summary of overall results for all subsets =====")
    print(f"Total subsets: {num_subsets}")
    print(f"Average accuracy across all subsets: {avg_metrics['accuracy']:.3f} ± {np.std(all_metrics['accuracy']):.3f}")
    print(f"Average accuracy of all subsets: {avg_metrics['precision']:.3f} ± {np.std(all_metrics['precision']):.3f}")
    print(f"Average recall of all subsets: {avg_metrics['recall']:.3f} ± {np.std(all_metrics['recall']):.3f}")
    print(f"Average F1-score across all subsets: {avg_metrics['f1']:.3f} ± {np.std(all_metrics['f1']):.3f}")
    print(f"Average AUC for all subsets: {avg_metrics['auc']:.3f} ± {np.std(all_metrics['auc']):.3f}")  # 汇总显示AUC

if __name__ == '__main__':
    main()