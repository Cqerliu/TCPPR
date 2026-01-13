import gc
import matplotlib.pyplot as plt
import os
import shap
import torch
import numpy as np
from encode import ProEncoder

# ---------------------- Multi-subset data de-redundancy function ----------------------
def merge_and_deduplicate_subsets(all_subset_data, all_subset_labels, all_subset_rnap_seqs, device):
    """
    Merge and deduplicate RNAP feature data from all subsets
    :param all_subset_data: List of RNAP features from each subset (torch.Tensor, shape=(batch, channels, length))
    :param all_subset_labels: List of labels from each subset (numpy.ndarray)
    :param all_subset_rnap_seqs: List of RNAP sequences from each subset (list)
    :param device: Device (CPU/GPU)
    :return: Deduplicated features, labels, sequences
    """
    merged_data = torch.cat(all_subset_data, dim=0).cpu().numpy()
    merged_labels = np.concatenate(all_subset_labels, axis=0)
    merged_seqs = []
    for seq_list in all_subset_rnap_seqs:
        merged_seqs.extend(seq_list)

    seq_to_idx = {}
    unique_indices = []
    for idx, seq in enumerate(merged_seqs):
        if seq not in seq_to_idx:
            seq_to_idx[seq] = idx
            unique_indices.append(idx)

    unique_data = merged_data[unique_indices]
    unique_labels = merged_labels[unique_indices]
    unique_seqs = [merged_seqs[idx] for idx in unique_indices]
    unique_data_tensor = torch.from_numpy(unique_data).float().to(device)

    print(
        f"Subset merging and deduplication completed: Original total samples={len(merged_seqs)} → Deduplicated={len(unique_seqs)}")
    return unique_data_tensor, unique_labels, unique_seqs

# ---------------------- SHAP analysis ----------------------
def plot_shap_rnap_analysis_species_level(model, all_subset_data, all_subset_labels, all_subset_rnap_seqs,
                                          device, species_name, save_dir, data_type="Test"):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    # Step 1: Merge and deduplicate subsets
    unique_fc, unique_labels, unique_rnap_seqs = merge_and_deduplicate_subsets(
        all_subset_data, all_subset_labels, all_subset_rnap_seqs, device
    )
    total_test_samples = len(unique_fc)
    channels = unique_fc.shape[1]
    length = unique_fc.shape[2]
    print(
        f"Deduplicated {data_type} samples: {total_test_samples}, RNAP feature shape: {unique_fc.shape} (batch, channels, length)")

    # Step 2: Dynamically adjust background/explain data size
    if total_test_samples <= 10:
        background_size = total_test_samples
        explain_size = total_test_samples
        if total_test_samples < 5:
            noise = torch.normal(0, 0.01, size=(5 - total_test_samples, channels, length)).to(device)
            unique_fc = torch.cat([unique_fc, noise], dim=0)
            total_test_samples = 5
            background_size = 5
            explain_size = 5
    elif total_test_samples <= 30:
        background_size = total_test_samples
        explain_size = total_test_samples
    elif total_test_samples <= 50:
        background_size = int(total_test_samples * 0.8)
        explain_size = total_test_samples
    else:
        background_size = 50
        explain_size = 100

    # Step 3: Sample background/explain data (keep 3D format)
    background_indices = np.random.choice(total_test_samples, size=background_size, replace=False)
    background_fc = unique_fc[background_indices].to(device)
    background_fc.requires_grad = True
    explain_indices = np.random.choice(total_test_samples, size=explain_size, replace=total_test_samples < explain_size)
    explain_fc = unique_fc[explain_indices].to(device)
    explain_fc.requires_grad = True
    print(f"SHAP analysis configuration ({data_type}): Background size={background_size}, Explain size={explain_size}")

    # Step 4: Model wrapping (consistent with training input format)
    class RNAPModel(torch.nn.Module):
        def __init__(self, cnn, mlp):
            super().__init__()
            self.cnn = cnn
            self.mlp = mlp

        def forward(self, x):
            features = self.cnn(x)
            logits = self.mlp(features)
            return logits[:, 1]  # Output positive class logit only

    wrapped_model = RNAPModel(model['cnn'], model['mlp']).to(device)
    wrapped_model.eval()
    local_smoothing = 0.05 if background_size < 20 else 0.01
    explainer = shap.GradientExplainer(model=wrapped_model, data=background_fc, local_smoothing=local_smoothing)

    # Step 5: Calculate SHAP values (extract positive class)
    shap_values = explainer.shap_values(explain_fc)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_pos = shap_values[1]
    elif shap_values.ndim == 4 and shap_values.shape[-1] == 2:
        shap_values_pos = shap_values[..., 1]
    else:
        shap_values_pos = shap_values
    print(
        f"{data_type} SHAP values shape: {shap_values.shape if isinstance(shap_values, np.ndarray) else [v.shape for v in shap_values]}")
    print(f"{data_type} Positive class SHAP values shape: {shap_values_pos.shape}")

    # Flatten to 2D
    shap_values_2d = shap_values_pos.reshape(shap_values_pos.shape[0], -1)
    explain_fc_2d = explain_fc.detach().cpu().numpy().reshape(explain_fc.shape[0], -1)
    print(
        f"{data_type} Flattened SHAP values shape: {shap_values_2d.shape}, Flattened input features shape: {explain_fc_2d.shape}")
    assert shap_values_2d.shape[1] == explain_fc_2d.shape[1], "Mismatched dimensions after flattening!"

    # ---------------------- Simplified mapping: Only show encoded sequences ----------------------
    def get_kmer_type_and_window(length_idx):
        """Determine k-mer type and window start position"""
        k1 = 7  # 1mer dimension (AIYHRDC)
        k2 = 7 ** 2  # 2mer dimension
        k3 = 7 ** 3  # 3mer dimension

        if length_idx < k1:
            return 1, length_idx
        elif length_idx < k1 + k2:
            return 2, length_idx - k1
        else:
            return 3, length_idx - (k1 + k2)

    def map_rnap_feature_simple(channel_idx, length_idx, rnap_seq):
        """Simplified feature name: Only channel and encoded fragment"""
        # Sequence preprocessing (clustering only, no original amino acid restoration)
        transtable = str.maketrans(ProEncoder.pro_intab, ProEncoder.pro_outtab)
        processed_seq = rnap_seq.translate(transtable)
        seq_len = len(processed_seq)
        if seq_len == 0:
            return f"Channel{channel_idx}_invalid_seq"

        # k-mer information (only for locating encoded fragment, not shown in name)
        kmer_type, window_start = get_kmer_type_and_window(length_idx)
        window_end = window_start + kmer_type
        if window_end > seq_len:
            window_start = max(0, seq_len - kmer_type)
            window_end = window_start + kmer_type
        encoded_fragment = processed_seq[window_start:window_end] if window_start < window_end else "invalid_pos"

        # Only keep channel and encoded fragment
        feature_name = f"Channel{channel_idx}_{encoded_fragment}"
        return feature_name

    # Use first unique sequence as mapping reference
    reference_rnap_seq = unique_rnap_seqs[0] if len(unique_rnap_seqs) > 0 else ""
    processed_ref_len = len(reference_rnap_seq.translate(str.maketrans(ProEncoder.pro_intab, ProEncoder.pro_outtab)))
    print(f"\n{data_type} Reference RNAP encoded sequence length: {processed_ref_len}")

    # Generate simplified feature names
    feature_names = []
    for flat_idx in range(shap_values_2d.shape[1]):
        channel_idx = flat_idx // length
        length_idx = flat_idx % length
        feat_name = map_rnap_feature_simple(channel_idx, length_idx, reference_rnap_seq)
        feature_names.append(feat_name)

    # ---------------------- Filter top 30 high-contribution features ----------------------
    feature_importance = np.mean(np.abs(shap_values_2d), axis=0)
    top30_feat_indices = np.argsort(feature_importance)[-30:][::-1]
    top30_shap_values = shap_values_2d[:, top30_feat_indices]
    top30_explain_fc = explain_fc_2d[:, top30_feat_indices]
    top30_feature_names = [feature_names[idx] for idx in top30_feat_indices]

    # Step 6: Plot SHAP summary plot 
    plt.figure(figsize=(14, 8))
    shap.summary_plot(
        top30_shap_values,
        top30_explain_fc,
        plot_type="dot",
        feature_names=top30_feature_names,
        show=False,
        max_display=30
    )

    plt.xlabel("SHAP Value", fontsize=12)
    plt.ylabel("RNAP Feature (Channel + Encoded Fragment)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    save_path = f"{save_dir}/shap_rnap_summary_{species_name}_{data_type.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{data_type} SHAP plot saved to: {save_path}")

    # Output top30 features 
    print(f"\nTop 30 High-Contribution RNAP Features ({species_name} - {data_type} Set):")
    for rank, (feat_idx, shap_idx) in enumerate(zip(top30_feat_indices, range(20)), 1):
        channel_idx = feat_idx // length
        length_idx = feat_idx % length
        feat_detail = top30_feature_names[shap_idx]
        importance_score = feature_importance[feat_idx]
        print(f"\n  Rank {rank}: {feat_detail} | Average Importance Score: {importance_score:.4f}")

    # Channel distribution 
    channel_top30_count = {}
    for feat_idx in top30_feat_indices:
        channel_idx = feat_idx // length
        channel_top30_count[channel_idx] = channel_top30_count.get(channel_idx, 0) + 1
    print(f"\nChannel Distribution in Top 20 Features ({data_type} Set):")
    for channel_idx in sorted(channel_top30_count.keys()):
        count = channel_top30_count[channel_idx]
        print(f"  Channel {channel_idx}: {count} features ({count / 20 * 100:.1f}%)")

    if total_test_samples <= 30:
        print(
            f"\nNote: Small deduplicated {data_type} set ({total_test_samples} samples), SHAP results are for reference only")



# ---------------------- Multiple sampling to calculate Promoter and RNAP contribution scores (averaged) ----------------------
def calculate_contribution_scores(
        model_dict, x_data, fc_data, device, n_samples=5):
    prom_scores = []
    rnap_scores = []

    for sample_idx in range(n_samples):
        # Use different random seeds for each sample
        np.random.seed(42 + sample_idx)
        torch.manual_seed(42 + sample_idx)

        prom_score, rnap_score = _calculate_single_contribution(
            model_dict, x_data, fc_data, device)

        prom_scores.append(prom_score)
        rnap_scores.append(rnap_score)

    avg_prom = np.mean(prom_scores)
    avg_rnap = np.mean(rnap_scores)

    return {
        'avg_promoter': avg_prom,'avg_rnap': avg_rnap,
        'all_promoter': prom_scores,'all_rnap': rnap_scores
    }

# ---------------------- Calculation of single contribution ----------------------
def _calculate_single_contribution(model_dict, x_data, fc_data, device):
    for sub_model in model_dict.values():
        if hasattr(sub_model, 'eval'):
            sub_model.eval()

    x_tensor = torch.from_numpy(x_data).float().to(device)
    fc_tensor = torch.from_numpy(fc_data).float().to(device) if fc_data is not None else None

    with torch.no_grad():
        promoter_features = model_dict['transformer'](x_tensor)
        if fc_tensor is not None and 'cnn' in model_dict:
            rnap_features = model_dict['cnn'](fc_tensor)
            fusion_features, _, _ = model_dict['fusion'](promoter_features, rnap_features)
        else:
            fusion_features, _, _ = model_dict['fusion'](promoter_features)
        original_probs = torch.softmax(model_dict['mlp'](fusion_features), dim=1)[:, 1].cpu().numpy()

        # Promoter forecast only
        if fc_tensor is not None and 'cnn' in model_dict:
            zero_rnap = torch.zeros_like(fc_tensor)
            rnap_features_zero = model_dict['cnn'](zero_rnap)
            fusion_prom_only, _, _ = model_dict['fusion'](promoter_features, rnap_features_zero)
        else:
            fusion_prom_only, _, _ = model_dict['fusion'](promoter_features)
        prom_only_probs = torch.softmax(model_dict['mlp'](fusion_prom_only), dim=1)[:, 1].cpu().numpy()

        # RNAP prediction only
        zero_promoter = torch.zeros_like(x_tensor)
        promoter_features_zero = model_dict['transformer'](zero_promoter)
        if fc_tensor is not None and 'cnn' in model_dict:
            fusion_rnap_only, _, _ = model_dict['fusion'](promoter_features_zero, rnap_features)
        else:
            fusion_rnap_only, _, _ = model_dict['fusion'](promoter_features_zero)
        rnap_only_probs = torch.softmax(model_dict['mlp'](fusion_rnap_only), dim=1)[:, 1].cpu().numpy()

    prom_contribution = np.mean(np.abs(original_probs - rnap_only_probs))
    rnap_contribution = np.mean(np.abs(original_probs - prom_only_probs))

    return prom_contribution, rnap_contribution

# ---------------------- Comparative mapping of contributions ----------------------
def plot_aesthetic_donut(prom_score, rnap_score, species_name, save_dir):
    total = prom_score + rnap_score
    prom_pct = (prom_score / total) * 100
    rnap_pct = (rnap_score / total) * 100
    labels = ['Promoter DNA', 'RNAP Context']
    sizes = [prom_score, rnap_score]
    colors = ['#4e79a7', '#f28e2b'] 

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    wedges, texts, autotexts = ax.pie(
        sizes,labels=labels,autopct='%1.1f%%',
        startangle=90,pctdistance=0.85, colors=colors,
        textprops={'fontsize': 12, 'weight': 'bold', 'color': '#333333'},
        wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 2}, 
        explode=(0.03, 0.03) )

    for text in texts:
        text.set_fontsize(13)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)

    plt.text(0, 0, f'{species_name}\nModality\nContribution',
             ha='center', va='center', fontsize=14, fontweight='bold', color='#555555',
             style='italic')  

    title_text = f'Feature Contribution Analysis ({species_name})'
    plt.title(title_text, fontsize=16, pad=20, style='italic')  

    ax.legend(wedges, [f"{l} ({s / total:.1%})" for l, s in zip(labels, sizes)],
              title="Modalities", loc="center left",
              bbox_to_anchor=(1, -0.3, 0.5, 1))

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{species_name}_modality_contribution_donut.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"The contribution loop has been saved to: {save_path}")