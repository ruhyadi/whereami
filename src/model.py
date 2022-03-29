"""
script for model generation
"""
from typing import Any, List

import torch
from torchmetrics import MaxMetric
import torchvision
from torch import nn
from torchvision import models

from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

class LitModel(LightningModule):
    def __init__(
        self, 
        model: torch.nn.Module, 
        learning_rate: float = 0.001, 
        batch_size: int = 32
        ):
        super().__init__()
        # save hyperparamters
        self.save_hyperparameters(logger=False)
        # model arc
        self.model = model
        # loss function
        self.criterion = torch.nn.MSELoss()

        # accuracy metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        x = x.float()
        y = y.float()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(y.squeeze(), dim=1)
        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss from training_step
        return {'loss': loss, 'preds': preds, 'targets': targets}

    def training_step_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute() # get val accuracy
        self.val_acc_best.update(acc)
        self.log('val/acc_best', self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.learning_rate
        )
        return [optimizer]
    

class Model(nn.Module):
    def __init__(self, backbone, class_):
        super(Model, self).__init__()
        self.select_bakcbone = backbone
        self.num_class = len(class_)
        self.model = self.backbone(self.select_bakcbone)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_class)
        
    def forward(self, x):
        x = self.model(x)
        return x

    def backbone(self, select_backbone):
        model = {
            'resnet18': models.resnet18(pretrained=True),
            'resnet50': models.resnet50(pretrained=True)
        }
        return model[select_backbone]

if __name__ == '__main__':
    
    from torchvision import models

    resnet18 = models.resnet18()
    resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 3)

    resnet50 = models.resnet50()

    for net in [resnet18, resnet50]:
        model = LitModel(net)

        print(model)