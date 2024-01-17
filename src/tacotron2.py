import torch
import torch.nn as nn

from src.modules.encoder import Encoder
from src.modules.decoder import Decoder
from src.utils.net import PostNet
from src.utils.masking import generate_mask

from typing import Optional

class Tacotron2(nn.Module):
    def __init__(self,
                token_size: int,
                n_mel_channels: int,
                embedding_dim: int,
                encoder_kernel_size: int,
                attention_dim: int,
                attention_rnn_dim: int,
                decoder_rnn_dim: int,
                n_filters: int,
                prenet_dim: int,
                location_kernel_size: int,
                encoder_n_convolutions: int,
                postnet_n_convolutions: int,
                postnet_kernel_size: int,
                dropout_rate: float) -> None:
        super().__init__()
        self.encoder = Encoder(
            token_size=token_size,
            embedding_dim=embedding_dim,
            n_convolutions=encoder_n_convolutions,
            kernel_size=encoder_kernel_size,
            dropout_rate=dropout_rate
        )

        self.decoder = Decoder(
            n_mel_channels=n_mel_channels,
            attention_rnn_dim=attention_rnn_dim,
            attention_dim=attention_dim,
            embedding_dim=embedding_dim,
            prenet_dim=prenet_dim,
            decoder_rnn_dim=decoder_rnn_dim,
            n_filters=n_filters,
            kernel_size=location_kernel_size,
            dropout_rate=dropout_rate
        )

        self.postnet = PostNet(
            n_convolutions=postnet_n_convolutions,
            n_mel_channels=n_mel_channels,
            embedding_dim=embedding_dim,
            kernel_size=postnet_kernel_size,
            dropout_rate=dropout_rate
        )

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        mask = None
        if lengths is not None:
            mask = generate_mask(lengths)
        encoder_outputs = self.encoder(encoder_inputs, lengths)

        mel_outputs, gate_outputs = self.decoder(encoder_outputs, decoder_inputs, mask)
        postnet_mel_outputs = self.postnet(mel_outputs)
        postnet_mel_outputs += mel_outputs

        return postnet_mel_outputs, mel_outputs, gate_outputs
    
    def inference(self, encoder_inputs: torch.Tensor, max_decoder_steps: int, gate_threshold: float = 0.5):
        encoder_outputs = self.encoder.inference(encoder_inputs)

        mel_outputs, _ = self.decoder.inference(encoder_outputs, max_decoder_steps, gate_threshold)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs_postnet + mel_outputs

        return mel_outputs_postnet