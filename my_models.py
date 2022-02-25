from abc import ABCMeta, abstractmethod
import pytorch_lightning as pl
from torch import Tensor, nn
import torch
from torch.nn.functional import cross_entropy, mse_loss
from torchmetrics import functional as metrics

def flip_base_module(module: nn.Module):
    if type(module) is nn.Linear:
        return nn.Linear(in_features=module.out_features, out_features=module.in_features)
    elif type(module) is nn.ReLU:
        return nn.ReLU()


def generate_reverse_module(module: nn.Module):
    children = module.children()
    if len(children) == 0:
        return flip_base_module(module)
    return [generate_reverse_module(children_module) for children_module in children[::-1]]


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size, encoding_size, num_labels):
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size,
                      out_features=encoding_size),
            nn.ReLU(),
            nn.Linear(in_features=encoding_size, out_features=num_labels)
        )
        self.decoder = generate_reverse_module(self.encoder)

    def forward(self, x: Tensor):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder.forward(x)
        x_decoded = self.decoder(x_encoded)
        loss = mse_loss(x_decoded, x) + cross_entropy(x_encoded, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_encoded = self.encoder.forward(x)
        y_hat = torch.argmax(x_encoded, dim=1)
        self.log({'val_ce': cross_entropy(x_encoded, y), 'val_f1': metrics.f1(y_hat, y)})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
