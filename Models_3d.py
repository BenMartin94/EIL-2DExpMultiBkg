import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EncoderBlock(nn.Module):
    """
    A PyTorch recreation of the Keras EncoderBlock.
    This block consists of two convolutional layers followed by a max-pooling layer.
    It implements a skip connection by returning the output of the second conv layer
    before the pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float):
        super().__init__()
        # Keras 'padding=same' with a 5x5 kernel is equivalent to padding=2 in PyTorch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.conv1(x))
        skip = self.relu(self.conv2(x)) # This will be used for the skip connection
        x = self.maxpool(skip)
        x = self.dropout(x)
        return x, skip

class DecoderBlock(nn.Module):
    """
    A PyTorch recreation of the Keras DecoderBlock.
    This block upsamples the feature map and concatenates it with a skip connection
    from the corresponding encoder block.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float):
        super().__init__()
        # This ConvTranspose2d is configured to double the height and width of the input tensor
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        # The next conv layer will have doubled input channels because of the skip connection
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.convtrans(x)
        # Concatenate along the channel dimension (dim=1)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv(x))
        x = self.dropout(x)
        return x

class UnetModel(nn.Module):
    """
    A PyTorch recreation of the Keras UnetModel.
    This model uses the EncoderBlock and DecoderBlock to form a U-Net architecture.
    The final layers are designed to produce a 3D volumetric output from a 2D-like input
    by shaping the output channels to represent the depth dimension.
    """
    def __init__(self, in_channels: int, image_dim: int, dropout_rate: float = 0.0, final_activation: str = 'linear'):
        super().__init__()
        self.image_dim = image_dim

        self.encode1 = EncoderBlock(in_channels, 16, dropout_rate)
        self.encode2 = EncoderBlock(16, 32, dropout_rate)

        self.bottleneck1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bottleneck2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.decode1 = DecoderBlock(64, 32, dropout_rate)
        self.decode2 = DecoderBlock(32, 16, dropout_rate)

        self.conv1 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, self.image_dim // 2, kernel_size=1)
        # This final conv layer creates image_dim channels, which will serve as the depth of the output volume
        self.conv3 = nn.Conv2d(self.image_dim // 2, self.image_dim, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # The original Keras code doesn't specify a final activation, so we default to linear (no-op)
        if final_activation == 'linear':
            self.final_activation = nn.Identity()
        else:
            # You can add other activations here if needed, e.g., nn.Sigmoid()
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1, skip1 = self.encode1(x)
        e2, skip2 = self.encode2(e1)

        x = self.relu(self.bottleneck1(e2))
        x = self.relu(self.bottleneck2(x))

        d1 = self.decode1(x, skip2)
        d2 = self.decode2(d1, skip1)

        x = self.relu(self.conv1(d2))
        x = self.relu(self.conv2(x))
        x = self.conv3(x) # No relu here to match Keras model structure

        # This is the "2.5D" trick: upsample the H and W dimensions to match the target image_dim.
        # The channel dimension from the previous layer now represents the depth.
        # The output shape will be (N, depth, height, width), e.g., (batch, 56, 56, 56)
        x = F.interpolate(x, size=(self.image_dim, self.image_dim), mode='bilinear', align_corners=False)
        
        x = self.final_activation(x)

        return x

class TransformerUNet(nn.Module):
    """
    The main model, combining a Transformer encoder with the U-Net decoder.
    This model processes sequential frequency data, reshapes it into a 2D-like
    feature map, and then uses the U-Net to reconstruct a 3D volume.
    """
    def __init__(self, num_transformer_layers: int, num_heads: int,
                 num_receivers: int = 24, num_sources: int = 24, num_freqs: int = 10, image_dim: int = 56):
        super().__init__()
        self.num_receivers = num_receivers
        self.num_sources = num_sources
        self.num_freqs = num_freqs

        embed_dim = num_receivers * num_sources # In the Keras code, this is 576
        
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2, # A common choice for feedforward dim
                dropout=0.1,
                activation='relu',
                batch_first=True # This simplifies tensor manipulation
            ) for _ in range(num_transformer_layers)
        ])

        # U-Net Decoder
        # The number of input channels to the U-Net is the number of complex frequencies
        unet_in_channels = 2 * num_freqs
        self.unet = UnetModel(in_channels=unet_in_channels, image_dim=image_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to have shape (batch_size, seq_len, embed_dim)
        # e.g., (batch, 20, 576)

        # 1. Pass through Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # 2. Reshape for the U-Net
        # The Keras code reshapes and transposes. The equivalent here is a single view operation
        # to get the shape (batch_size, channels, height, width) for the PyTorch U-Net.
        # (batch, 20, 576) -> (batch, 20, 24, 24)
        x = x.view(-1, 2 * self.num_freqs, self.num_receivers, self.num_sources)

        # 3. Pass through U-Net to get the final 3D volume
        # Output shape: (batch, image_dim, image_dim, image_dim) e.g. (batch, 56, 56, 56)
        x = self.unet(x)
        
        return x

if __name__ == '__main__':
    # --- Configuration based on the original Keras scripts ---
    BATCH_SIZE = 4
    NUM_TRANSFORMER_LAYERS = 1
    NUM_HEADS = 16
    NUM_RECEIVERS = 24
    NUM_SOURCES = 24
    NUM_FREQS = 10
    IMAGE_DIM = 56 # The dimension of the output cube (56x56x56)

    # --- Instantiate the model ---
    model = TransformerUNet(
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        num_receivers=NUM_RECEIVERS,
        num_sources=NUM_SOURCES,
        num_freqs=NUM_FREQS,
        image_dim=IMAGE_DIM
    )

    # --- Create some dummy data to test the model ---
    # The input shape matches the pre-processing from train_transformer_unet.py
    # (batch_size, sequence_length, embedding_dimension)
    # sequence_length = 2 * num_freqs
    # embedding_dimension = num_receivers * num_sources
    dummy_input = torch.randn(BATCH_SIZE, 2 * NUM_FREQS, NUM_RECEIVERS * NUM_SOURCES)

    print(f"Input shape:  {dummy_input.shape}")

    # --- Forward pass ---
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({BATCH_SIZE}, {IMAGE_DIM}, {IMAGE_DIM}, {IMAGE_DIM})")

    # --- Check if the output shape is correct ---
    assert output.shape == (BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, IMAGE_DIM)
    print("\nModel instantiated and tested successfully!")

    # --- Print model summary ---
    try:
        from torchinfo import summary
        summary(model, input_size=dummy_input.shape)
    except ImportError:
        print("\nInstall 'torchinfo' (pip install torchinfo) for a detailed model summary.")
        print(model)
