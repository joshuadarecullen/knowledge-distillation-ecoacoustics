from typing import List, Dict, Any, Iterator, Union

import wandb
from torch import Tensor, cat
from torch.functional import F
import pytorch_lightning as pl
from torchmetrics import functional as M
from torchmetrics import ConfusionMatrix

from sklearn import metrics

from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics.functional import multiclass_accuracy
from torcheval.metrics.functional import mean_squared_error
from torcheval.metrics.functional import multiclass_precision
from torcheval.metrics.functional import multiclass_recall

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Colormap

from conduit.data import TernarySample
from conduit.data.datasets.audio.ecoacoustics import SoundscapeAttr

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.transforms.transforms import Framing
from src.utils.utils import generate_title_string
from src.utils.types import Split

class MLPMetrics(pl.Callback):
    def __init__(self, cmap: Union[str, Colormap] = 'Reds' ) -> None:
        self.cmap = cmap
        super().__init__()

    '''
    This callback calculates metrics for the multiclass mlp every train, val, and test and predict epochs 
    f1_score, accuracy and mse of categorical and continous targets respectively.
    '''

    def on_train_epoch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             ) -> None:
        # create train dictionary for storing target information at the start of every epoch.
        self.train_data = {k: [] for k in trainer.datamodule.target_attrs[:-1]}
        self.train_data['y'] = []

    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Tensor],
                           batch: TernarySample,
                           batch_idx: int) -> None:
        # loop through habitat metrics and store into train data
        for i, target in enumerate(trainer.datamodule.target_attrs[:-1]):
            self.train_data[target].append(outputs[target].detach().cpu())
        self.train_data['y'].append(outputs['y'].cpu())

    def on_train_epoch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          ) -> None:

        # end batches will not have the same amount of observations as previous batch, cat handles this for us
        train_data = {k: cat(v) for k, v in self.train_data.items()}
        # compute the metrics at the end of the epoch
        metrics = self.compute_metrics(trainer.datamodule.target_attrs[:-1], train_data, 'train')

        # log metrics to wandb
        pl_module.log("train/f1_score", metrics["train/f1_score"])
        pl_module.logger.experiment.log(metrics)


    def on_validation_epoch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             ) -> None:
        # same as train
        self.val_data = {k: [] for k in trainer.datamodule.target_attrs[:-1]}
        self.val_data['y'] = []

    def on_validation_batch_end(self,
                                trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: TernarySample,
                                batch_idx: int,
                                ) -> None:
        # same as train
        for i, target in enumerate(trainer.datamodule.target_attrs[:-1]):
            self.val_data[target].append(outputs[target].detach().cpu())
        self.val_data['y'].append(outputs['y'].cpu())

    def on_validation_epoch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          ) -> None:


        # same as train
        val_data = {k: cat(v) for k, v in self.val_data.items()}

        metrics = self.compute_metrics(trainer.datamodule.target_attrs[:-1], val_data, 'val')

        # log the metrics
        pl_module.log("val/f1_score", metrics["val/f1_score"])
        pl_module.logger.experiment.log(metrics)


    def on_test_epoch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             ) -> None:
        # same as train
        self.test_data = {k: [] for k in trainer.datamodule.target_attrs[:-1]}
        self.test_data['y'] = []

    def on_test_batch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          outputs: Dict[str, Tensor],
                          batch: TernarySample,
                          batch_idx: int) -> None:

        # same as train
        for i, target in enumerate(trainer.datamodule.target_attrs[:-1]):
            self.test_data[target].append(outputs[target].detach().cpu())

        self.test_data['y'].append(outputs['y'].cpu())

    def on_test_epoch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          ) -> None:

        test_data = {k: cat(v) for k, v in self.test_data.items()}
        # generate metrics
        metrics = self.compute_metrics(trainer.datamodule.target_attrs[:-1], test_data, 'test')
        # generate confusion matrix
        cm_fig = self.generate_confusion_matrix(target='habitat',
                                                f1_score= metrics['test/f1_score'],
                                                y_preds=test_data['habitat'],
                                                y_true=test_data['y'][:,2],
                                                decoder=trainer.datamodule.train_data.dataset.decoder,
                                                embedder=pl_module.embedder,
                                                split='test')

        # log metrics
        pl_module.log("test/f1_score", metrics["test/f1_score"])
        pl_module.logger.experiment.log({ f"test/confusion/habitat": wandb.Image(cm_fig) })
        plt.close(cm_fig)
        pl_module.logger.experiment.log(metrics)

    def on_predict_epoch_start(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule) -> None:
        # helps with the logic in on_predict_batch_start for the sequential loader
        self.predict_data = None

    def on_predict_batch_start(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          batch: TernarySample,
                          batch_idx: int,
                          dataloader_idx: int) -> None:

        # first dataloader is the training data, else its validation data
        split = 'train' if dataloader_idx == 0 else 'val'
        if not self.predict_data:
            self.predict_data = {split: {k: [] for k in trainer.datamodule.target_attrs[:-1]}}
            self.predict_data[split]['y'] = []
        elif len(self.predict_data.keys()) == 1 and split == 'val':
            self.predict_data[split] = {k: [] for k in trainer.datamodule.target_attrs[:-1]}
            self.predict_data[split]['y'] = []

    def on_predict_batch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          outputs: Dict[str, Tensor],
                          batch: TernarySample,
                          batch_idx: int,
                          dataloader_idx: int) -> None:

        split = 'train' if dataloader_idx == 0 else 'val'

        for i, target in enumerate(trainer.datamodule.target_attrs[:-1]):
            self.predict_data[split][target].append(outputs[split][target].detach().cpu())

        self.predict_data[split]['y'].append(outputs[split]['y'].cpu())

    def on_predict_epoch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule) -> None:

        # concat for both train and val data
        predict_data = {split: {k: cat(v) for k, v in vs.items()} for split, vs in self.predict_data.items()}

        for split, output in predict_data.items():

            metrics = self.compute_metrics(trainer.datamodule.target_attrs[:-1], output, split)
            cm_fig = self.generate_confusion_matrix(target='habitat',
                                                    f1_score= metrics[f'{split}/f1_score'],
                                                    y_preds=output['habitat'],
                                                    y_true=output['y'][:,2],
                                                    decoder=trainer.datamodule.train_data.dataset.decoder,
                                                    embedder=pl_module.embedder,
                                                    split=split)

           # Logging the metrics dict
            for key, value in metrics.items():
                pl_module.logger.experiment.log({f'{key}': value})

            pl_module.logger.experiment.log({ f"{split}/confusion/habitat": wandb.Image(cm_fig) })
            plt.close(cm_fig)

    def compute_metrics(self, targets: SoundscapeAttr, outputs: Dict[str, Tensor], split: str):
        # define a results dictionary for accuracy, f1 scores
        metrics= {}

        # define average scores
        num_continuous = 2 if 'NN' in outputs.keys() and 'N0' in outputs.keys() else 1

        # compute metrics for continous and categorical
        for i, (target) in enumerate(targets):
            y = outputs['y']
            if target in 'NN':
                mse_nn = mean_squared_error(outputs[target].flatten(), y[:,i]).item()
                metrics[f"{split}/mse/{target}"] = mse_nn
            elif target in 'N0':
                mse_n0 = mean_squared_error(outputs[target].flatten(), y[:,i]).item()
                metrics[f"{split}/mse/{target}"] = mse_n0
            elif target in 'habitat':
                _, y_pred = F.softmax(outputs[target], dim=1).max(axis=1)
                metrics[f"{split}/f1_score"] = multiclass_f1_score(y_pred, y[:,i], num_classes=6, average="macro").item()
                metrics[f"{split}/accuracy"] = multiclass_accuracy(y_pred, y[:,i], num_classes=6).item()
                metrics[f"{split}/precision"] = multiclass_precision(y_pred, y[:,i], num_classes=6, average='macro')
                metrics[f"{split}/recall"] = multiclass_recall(y_pred, y[:,i], num_classes=6, average='macro')

        if num_continuous == 2:
            metrics[f'{split}/error_avg'] = (metrics[f"{split}/mse/NN"] + metrics[f"{split}/mse/N0"]) / num_continuous

        return metrics

    # constructing a confusion matrix
    def generate_confusion_matrix(self,
                                  target: str,
                                  f1_score: float,
                                  y_preds: Tensor,
                                  y_true: Tensor,
                                  decoder: Dict[str, int],
                                  embedder: str,
                                  split: str) -> Figure:

        labels = list(decoder[target].keys())
        display_labels = list(decoder[target].values())

        # create a figure and 
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
        ax = fig.add_subplot(111)

        _, y_pred = F.softmax(y_preds, dim=1).max(axis=1)
        # plot the matrix
        metrics.ConfusionMatrixDisplay.from_predictions(y_pred,
                y_true,
                labels=labels,
                display_labels=display_labels,
                xticks_rotation=45,
                ax=ax,
                cmap=self.cmap)
        # set titles
        ax.set_title(f"F1 Score: {f1_score}\n"
                f"Split: {split}")
        fig.suptitle(f"{embedder} MLP Confusion Matrix")
        return fig
