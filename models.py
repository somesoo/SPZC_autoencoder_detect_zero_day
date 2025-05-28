import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: list = [64, 32, 16],
        decoder_dims: list = [32, 64],
        use_batchnorm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        decoder_dims = decoder_dims + [input_dim]  # zakończ na input_dim
        prev_dim = encoder_dims[-1]
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if dim != input_dim:  # ostatnia warstwa bez aktywacji
                decoder_layers.append(nn.ReLU())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
