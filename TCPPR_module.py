import torch
import torch.nn.functional as F
from torch import nn, einsum
from entmax import Sparsemax
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.selector = Sparsemax(dim=-1)
    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.selector(sim)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)

# extract features for promoter
class TransformerFeature(nn.Module):
    def __init__(self,
                 input_dim,
                 max_seq_len,
                 depth=3,
                 heads=8,
                 dim_head=16,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 use_projection=True,
                 pool_type='mean'
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.use_projection = use_projection
        self.pool_type = pool_type

        if use_projection:
            self.projection = nn.Linear(input_dim, input_dim)

        # learnable positional encoding
        self.pos_embeds = nn.Embedding(max_seq_len, input_dim)

        # If a cls token is used, add a dedicated embedding.
        if pool_type == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))  # (1,1,input_dim)

        # Transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(input_dim, Attention(
                    input_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout
                ))),
                Residual(PreNorm(input_dim, FeedForward(input_dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):
        """
        x: input vectors，shape: (B, seq_len, input_dim)
        global_feat: global feature vectors（shape：(B, input_dim)）
        """
        B, seq_len, _ = x.shape

        if self.use_projection:
            x = self.projection(x)

        if self.pool_type == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,input_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, seq_len+1, input_dim)
            seq_len += 1  # Update the sequence length (including the cls token)

        # position encoding
        pos_indices = torch.arange(seq_len, device=x.device)  # (seq_len,)
        pos_emb = self.pos_embeds(pos_indices).unsqueeze(0)  # (1, seq_len, input_dim)
        x = x + pos_emb  # (B, seq_len, input_dim)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        #  pooling
        if self.pool_type == 'mean':
            # Mean pooling: Take the average over the sequence length dimension.
            global_feat = x.mean(dim=1)  # (B, input_dim)
        elif self.pool_type == 'max':
            # Max pooling: Take the maximum value over the sequence length dimension.
            global_feat, _ = x.max(dim=1)  # (B, input_dim)
        elif self.pool_type == 'cls':
            # CLS token: Take the first element of the sequence（cls_token）
            global_feat = x[:, 0, :]  # (B, input_dim)
        else:
            raise ValueError(f"不支持的池化方式：{self.pool_type}")

        return global_feat
# -----------------------------------------------------------------------------------------------------------------------
# extract features for RNAP

class CNNFeature(nn.Module):
    """
    CNN-based trainable feature extractor that receives the encoding vector generated by ProEncoder and outputs the feature vector.
    """
    def __init__(self,
                 input_length,
                 input_channels=1,
                 feature_dim=128,
                 conv_filters=[32, 64],
                 conv_kernels=[3, 3],
                 pool_sizes=[2, 2],
                 dropout_rate=0.3):

        super().__init__()
        self.input_length = input_length
        self.feature_dim = feature_dim

        layers = []
        in_channels = input_channels

        for i in range(len(conv_filters)):
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_filters[i],
                kernel_size=conv_kernels[i],
                padding='same'  # Ensure the length remains unchanged after convolution.
            ))
            layers.append(nn.BatchNorm1d(conv_filters[i]))
            layers.append(nn.ReLU(inplace=True))

            # pooling
            layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i]))

            # Dropout layer
            layers.append(nn.Dropout(dropout_rate))

            in_channels = conv_filters[i]  # Update the number of input channels.

        # conv
        self.conv_layers = nn.Sequential(*layers)

        # Calculate the feature length after convolution (for the fully connected layer)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_length)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.view(1, -1).size(1)

        # Fully connected layer: Map convolutional features to the target dimension.
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, feature_dim)
        )

    def forward(self, x):
        """
        x: input tensor，shape: (batch_size, channels, length)
        features: extracted vectors，shape: (batch_size, feature_dim)
        """
        # conv
        conv_out = self.conv_layers(x)
        # flatten
        flatten = conv_out.view(conv_out.size(0), -1)
        # Fully connected layer mapped to the feature dimension.
        features = self.fc_layers(flatten)

        return features
# -----------------------------------------------------------------------------------------------------------------------
# fuse features
class FeatureFusion(nn.Module):
    """Feature fusion module:
    Concatenate two input features and refine them through a multi-layer perceptron to adapt to the classification task."""
    def __init__(self, feature_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.concat_dim = feature_dim * 2

        self.hidden_dim = hidden_dim if hidden_dim is not None else self.concat_dim // 2

        self.fusion_network = nn.Sequential(
            # Reduce dimensions from the concatenated dimension to the hidden layer dimension.
            nn.Linear(self.concat_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Further refine the features.
            nn.Linear(self.hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Maintain the same dimension as the input features to facilitate processing by subsequent classifiers.
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, ft, fc):
        """
        :param ft: promoter feature tensor，shape (batch_size, feature_dim)
        :param fc: RNAP feature tensor，shape (batch_size, feature_dim)
        :return: fused feature tensor，shape (batch_size, feature_dim)
        """
        if ft.shape[1] != self.feature_dim or fc.shape[1] != self.feature_dim:
            raise ValueError(
                f"输入特征维度不匹配！预期 {self.feature_dim}，实际 ft: {ft.shape[1]}, fc: {fc.shape[1]}"
            )

        #  Concatenate two input features along the feature dimension.
        concatenated_features = torch.cat([ft, fc], dim=1)

        # 2. Refine features through a fusion network and output feature vectors suitable for classification.
        # output shape：(batch_size, feature_dim)
        fused_features = self.fusion_network(concatenated_features)

        return fused_features
# -----------------------------------------------------------------------------------------------------------------------
# classification
class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dims=[128,64],
                 activation='relu',
                 dropout=0.2 ,
                 batch_norm=False
                 ):
        super().__init__()

        # Choose an activation function.
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: input features，shape (batch_size, input_dim)
        logits: Classification score，shape (batch_size, num_classes)
        """
        return self.mlp(x)