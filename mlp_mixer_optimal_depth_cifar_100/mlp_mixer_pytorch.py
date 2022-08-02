from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        num_hidden = embed_dim * num_heads
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(embed_dim, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

    def get_number_of_parameters(self):
        return 2 * self.num_heads * self.embed_dim ** 2


class TokenMixer(nn.Module):
    def __init__(self, embed_dim, num_patches, num_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(num_patches, num_heads, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, embed_dim)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, embed_dim, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, embed_dim)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, embed_dim, num_patches, num_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, num_heads, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, embed_dim)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, embed_dim)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, embed_dim, num_patches, num_heads, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_patches, embed_dim, num_heads, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_patches, embed_dim, num_heads, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, embed_dim)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, embed_dim)
        return x


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class MLPMixer(nn.Module):
    def __init__(
            self,
            image_size=256,
            patch_size=16,
            embed_dim=128,
            num_heads=2,
            depth=8,
            in_channels=3,
            num_classes=10,
            dropout=0.5,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        # per-patch fully-connected is equivalent to strided conv2d
        self.patcher = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.mixers = nn.Sequential(
            *[
                MixerLayer(num_patches, embed_dim, num_heads, dropout)
                for _ in range(depth)
            ]
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, embed_dim, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, embed_dim)
        # patches.shape == (batch_size, num_patches, embed_dim)
        embedding = self.mixers(patches)
        # embedding.shape == (batch_size, num_patches, embed_dim)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

    def get_block_cost(self):
        return 2 * MLP(self.embed_dim, self.num_heads, dropout=.0).get_number_of_parameters()

    def get_layer_cost(self):
        return MLP(self.embed_dim, self.num_heads, dropout=.0).get_number_of_parameters()
