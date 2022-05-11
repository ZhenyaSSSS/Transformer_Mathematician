import torch
import math
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
class PositionalEncoding(nn.Module):

    def __init__(self, 
                 d_model: int = 256, 
                 dropout: float = 0.1, 
                 step: int = 98,
                 max_len: int = 5000
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.step = step

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, structure: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            structure: Tensor, shape [seq_num], structure.sum() == seq_len
        """
        pe_temp = torch.stack([self.pe[i] + self.pe[count + self.step] if count > 0 else self.pe[i] for count, value in enumerate (structure) for i in range (value)])
        x = x + pe_temp
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, 
                 d_model: int = 256, 
                 vocab: int = 306
                 ):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        return (self.lut(x) * math.sqrt(self.d_model)).permute(1,0,2)

class Mathematician(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0001445439770745928,
        batch_size: int = 1024,
        d_model: int = 256,
        input_vocab: int = 306,
        nhead: int = 16,
        num_layers: int = 14,
        d_hid: int = 1024,
        dropout_encoder: float = 0.1,
        dropout_encoding: float = 0.1,
        num_input_token: int = 9,
        num_output_token: int = 4,
        output_vocab: int = 256,
        smoothing: float = 0.0,
        structure: Tensor = torch.tensor([1, 4, 4]),
        step_encoding: int = 98,
        **kwargs
    ):
        super(Mathematician, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.structure = structure
        self.token_embeddings = Embeddings(d_model, input_vocab) 
        self.positional_encoding = PositionalEncoding(d_model, dropout_encoding, step_encoding)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout_encoder), num_layers)    
        self.head = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(d_model * num_input_token, output_vocab * num_output_token), nn.Unflatten(1, (output_vocab, num_output_token)))
        self.loss_f = torch.nn.CrossEntropyLoss(label_smoothing = smoothing)
        self.save_hyperparameters()
    def forward(self, data, target):
      x = self.token_embeddings(data)
      x = self.positional_encoding(x, self.structure)
      x = self.transformer_encoder(x)
      return self.loss_f(self.head(x.permute(1, 2, 0)), target)
    def compute_logit(self, data):
      x = self.token_embeddings(data)
      x = self.positional_encoding(x, self.structure)
      x = self.transformer_encoder(x)
      return self.head(x.permute(1, 2, 0))
    def training_step(self, batch, batch_idx):
      data, target = batch
      loss = self(data, target)
      self.log("train_loss", loss)
      return loss
    def custom_histogram_adder(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
    def training_epoch_end(self, output):
      self.custom_histogram_adder()
    def train_dataloader(self):
      sampler = None
      # sampler = torch.utils.data.distributed.DistributedSampler(
      #         train_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)

      loader = DataLoader(train_data, sampler=sampler, batch_size=self.batch_size, drop_last=True)

      return loader
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) 
        return optimizer
