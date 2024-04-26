import torch
import torch.nn as nn

from torchmetrics.functional import accuracy
from torchmetrics import ConfusionMatrix

import pytorch_lightning as pl
from datetime import timedelta, datetime
import os
import csv

from .cnn import CNN

class FKGLightningModule(pl.LightningModule):
    def __init__(
        self
        , model_key
        , learning_rate
        , weight_decay
        , T_max
        , fc_input_num
        , backbone='cnn'
        , num_classes=3
        , in_channel=3
    ):   
        super(FKGLightningModule, self).__init__()
        
        self.model_key = model_key
        self.fc_input_num = fc_input_num
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.backbone = backbone
        
        self.model = self._load_model()
        self.criterion = nn.CrossEntropyLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        self.initiated_at = (datetime.now()+timedelta(hours=7)).strftime("%Y-%m-%d_%H-%M-%S")
        self.result_file = f'{model_key}_{self.initiated_at}.csv'
        
    def forward(self, x):
        return self.model(x)

    # https://github.com/Lightning-AI/pytorch-lightning/issues/3795
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters()
            , lr=self.learning_rate
            , weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer
            , T_max=self.T_max
        )
        
        return {
            "optimizer": optimizer
            , "lr_scheduler": {
                "scheduler": scheduler
                , "interval": "epoch"
                , "frequency": 1
            }
        }
    
    # https://github.com/Lightning-AI/lightning/discussions/17182
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean()

        # Calculate Recall and FPR
        logs = {
            'train/loss': avg_loss.to(torch.float32)
            , 'train/acc': avg_acc.to(torch.float32)
            , 'step': self.current_epoch
        }

        self.log_dict(logs, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self._save_logs_to_csv(avg_loss, avg_acc, None, None, 'train') # Recall and FPR are not calculated when training.
        self.training_step_outputs.clear()

    # https://github.com/Lightning-AI/lightning/discussions/17182
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
        
        cm = self.confusion_matrix.compute().float()
        recall, fpr = self._get_metrics(cm)

        logs = {
            'val/loss': avg_loss.to(torch.float32)
            , 'val/acc': avg_acc.to(torch.float32)
            , 'val/recall': recall.to(torch.float32)
            , 'val/fpr': fpr.to(torch.float32)
            , 'step': self.current_epoch
        }

        self.log_dict(logs, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self._save_logs_to_csv(avg_loss, avg_acc, recall, fpr, 'val')
        self.validation_step_outputs.clear()
        
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'validation')
    
    # -- Supporting functions -- #  
    def _load_model(self):
        if self.backbone == 'cnn':
            return CNN(
                model_key=self.model_key
                , fc_input_num=self.fc_input_num
                , num_classes=self.num_classes
                , in_channel=self.in_channel
            )
        else:
            # TabNN
            return None
    
    def _step(self, batch, batch_idx, split):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self(inputs)    
        loss = self.criterion(outputs, labels)
        acc = accuracy(outputs, labels, task='multiclass', num_classes=self.num_classes)
        
        metrics = {
            'loss': loss
            , 'acc': acc
        }
        
        if split == 'train': 
            self.training_step_outputs.append(metrics)
        if split == 'validation': 
            self.validation_step_outputs.append(metrics)
            self.confusion_matrix.update(outputs, labels)
            
        return metrics
    
    def _get_metrics(self, cm):
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        recall = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.0)
        fpr = fp / (fp + tn) if fp + tn > 0 else torch.tensor(0.0)

        return recall, fpr
    
    def _save_logs_to_csv(self, loss, acc, recall, fpr, split):
        filename = f'logs/{self.initiated_at}/{split}_{self.result_file}'
        file_exists = os.path.exists(filename)
        
        if not file_exists:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['loss', 'acc', 'recall', 'fpr'])
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if recall is None:
                writer.writerow([loss.item(), acc.item(), 0, 0])
            else:
                writer.writerow([loss.item(), acc.item(), recall.item(), fpr.item()])