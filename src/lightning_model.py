from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torch import nn
from torch.nn import functional as f
import torch


class LightningModel(LightningModule):
    def __init__(self, model, train_ds=None, val_ds=None):
        super().__init__()
        self.model = model
        self.metrics = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.train_ds = train_ds
        self.val_ds = val_ds

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):
        image, label = batch
        output = self.forward(image)
        loss = nn.CrossEntropyLoss()(output, label)
        accuracy = self.metrics(output, label)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        output = self.forward(image)
        loss = nn.CrossEntropyLoss()(output, label)
        accuracy = self.metrics(output, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True)

        return loss

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=32, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=32, shuffle=False, num_workers=2)


class LightningModelDistill(LightningModel):
    def __init__(self, teacher_model, student_model, train_ds, val_ds, temp, alpha):
        super().__init__(model=student_model, train_ds=train_ds, val_ds=val_ds)
        self.teacher_model = teacher_model
        self.temp = temp
        self.alpha = alpha

    def forward(self, x):
        teacher_output = self.teacher_model(x)
        student_output = self.model(x)
        return teacher_output, student_output

    def training_step(self, batch, batch_idx):
        image, label = batch
        teacher_output, output = self.forward(image)
        loss = nn.KLDivLoss()(
            f.log_softmax(output / self.temp, dim=1),
            f.softmax(teacher_output / self.temp, dim=1)
        ) * (self.alpha * self.temp * self.temp) + f.cross_entropy(output, label) * (1. - self.alpha)
        train_accuracy = self.metrics(output, label)
        distill_acc = self.metrics(output, teacher_output)
        self.log("Distill_train_loss", loss, on_step=False, on_epoch=True)
        self.log("Distill_train_acc", distill_acc, on_step=False, on_epoch=True)
        self.log("train_acc", train_accuracy, on_step=False, on_epoch=True)
        return loss
