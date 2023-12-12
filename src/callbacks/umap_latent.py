from typing import Callable, List, Dict, Tuple, Union, Any, Iterator
from numpy import typing as npt
from pathlib import Path

import umap

import torch
import pytorch_lightning as pl
import numpy as np
import wandb

import pandas as pd
from matplotlib.figure import Figure
from matplotlib import colors
from matplotlib import cm
from conduit.data.datasets.audio.ecoacoustics import SoundscapeAttr
from conduit.data import TernarySample

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.types import Split

class UMAPLatent(pl.Callback):
    def __init__(self,
                 save_path: Union[Path, None] = None,
                 metric: str = 'euclidean',
                 min_distance: int = 0,
                 num_neighbours: int = 50,
                 seed: int = 0,
                 num_components: int = 2,
                 model_name: str = 'VAE',
                 cmap: colors.Colormap = 'rainbow') -> None:

        super().__init__()

        self.save_path = save_path
        self.metric = metric
        self.min_distance = min_distance
        self.num_neighbours = num_neighbours
        self.seed = seed
        self.num_components = num_components
        self.model_name = model_name
        self.cmap = cmap
        self.test_data: Dict[str, np.ndarray]
        self.val_data: Dict[str, np.ndarray]
        self.predict_data: Dict[str, Dict[str, np.ndarray]]

    '''
    def on_validation_epoch_start(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          ) -> None:
        self.val_data= {'z': [], 'y': [], 's': []}

    def on_validation_batch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          outputs: Dict[str, torch.Tensor],
                          batch: TernarySample,
                          batch_idx: int
                          ) -> None:

        # store data from each batch
        self.val_data['z'].append(outputs['z'].cpu().numpy())
        self.val_data['y'].append(outputs['y'].cpu().numpy())
        self.val_data['s'].append(outputs['s'].cpu().numpy())

    def on_validation_epoch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          ) -> None:

        from matplotlib import pyplot as plt

        #collate and process all batches data
        zs = np.concatenate(self.val_data['z'])
        ys = np.concatenate(self.val_data['y'])
        ss = np.concatenate(self.val_data['s'])

        # generate figure for each target label
        for fig in self.generate_figures(targets=trainer.datamodule.target_attrs,
                                         decoder=trainer.datamodule.train_data.dataset.decoder,
                                         zs=zs,
                                         ys=ys,
                                         ss=ss,
                                         split='val'):

            pl_module.logger.experiment.log({ 'val/umap': wandb.Image(fig) })

            plt.close(fig)
    '''

    def on_test_epoch_start(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          ) -> None:
        self.test_data = {'z': [], 'y': [], 's': []}

    def on_test_batch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          outputs: Dict[str, torch.Tensor],
                          batch: TernarySample,
                          batch_idx: int
                          ) -> None:

        # store data from each batch
        self.test_data['z'].append(outputs['z'].cpu().numpy())
        self.test_data['y'].append(outputs['y'].cpu().numpy())
        self.test_data['s'].append(outputs['s'].cpu().numpy())

    def on_test_epoch_end(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          ) -> None:

        from matplotlib import pyplot as plt

        #collate and process all batches data
        zs = np.concatenate(self.test_data['z'])
        ys = np.concatenate(self.test_data['y'])
        ss = np.concatenate(self.test_data['s'])

        # generate figure for each target label
        for fig in self.generate_figures(targets=trainer.datamodule.target_attrs,
                                         decoder=trainer.datamodule.train_data.dataset.decoder,
                                         zs=zs,
                                         ys=ys,
                                         ss=ss,
                                         split='test'):

            pl_module.logger.experiment.log({ 'test/umap': wandb.Image(fig) })

        plt.close(fig)

    def on_predict_epoch_start(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule) -> None:
        self.predict_data = None

    def on_predict_batch_start(self,
                          trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          batch: TernarySample,
                          batch_idx: int,
                          dataloader_idx: int) -> None:

        split = 'train' if dataloader_idx == 0 else 'val'
        if not self.predict_data and split == 'train':
            self.predict_data = {split: {'z': [], 'y': [], 's': []}}#{k: [] for k in trainer.datamodule.target_attrs[:-1]}}
            self.predict_data[split]['y'] = []
        elif len(self.predict_data.keys()) == 1 and split == 'val':
            self.predict_data[split] = {'z': [], 'y': [], 's': []}
            self.predict_data[split]['y'] = []


    def on_predict_batch_end(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             outputs: Dict[Split, Dict[str, torch.Tensor]],
                             batch: TernarySample,
                             batch_idx: int,
                             dataloader_idx: int) -> None:

        split = 'train' if dataloader_idx == 0 else 'val'

        # store data for each splits batch
        self.predict_data[split]['z'].append(outputs[split]['z'].cpu().numpy())
        self.predict_data[split]['y'].append(outputs[split]['y'].cpu().numpy())
        self.predict_data[split]['s'].append(outputs[split]['s'].cpu().numpy())


    def on_predict_epoch_end(self,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule) -> None:

        from matplotlib import pyplot as plt

        for _, (split, output) in enumerate(self.predict_data.items()):

            #collate and process each splits batches
            zs = np.concatenate(output['z'])

            ys = np.concatenate(output['y'])

            ss = np.concatenate(output['s'])

            # generate figures for each split
            for fig in self.generate_figures(targets=trainer.datamodule.target_attrs,
                                             decoder=trainer.datamodule.train_data.dataset.decoder,
                                             zs=zs,
                                             ys=ys,
                                             ss=ss,
                                             split=split):

                pl_module.logger.experiment.log({ f'{split}/umap': wandb.Image(fig) })
            plt.close(fig)


    def generate_figures(self,
                         targets: List[SoundscapeAttr],
                         decoder: Dict[int, str],
                         zs: npt.NDArray,
                         ys: npt.NDArray,
                         ss: npt.NDArray,
                         split: str=None) -> Figure:
        # Args:
        # decoder: label decoder 
        # targets: the list of targets
        # zs: tensor of embeddings [n, 128]
        # ys: ground truth
        # ss: dataset index

        # for sum reason during test a psutil error kept emerging, this stopped it
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        mapper = umap.UMAP(metric=self.metric,
                           min_dist=self.min_distance,
                           n_neighbors=self.num_neighbours,
                           random_state=self.seed,
                           n_components=self.num_components)

        # fit umap with latent variables
        embedding = mapper.fit_transform(zs)

        # emunerate targets labels except site
        for i , target_attr in enumerate(targets[:-1]):


            fig = plt.figure(figsize=(11.69, 8.27), dpi=100)

            # Continuous Targets
            if target_attr in 'NN' or target_attr in 'N0':

                # create a scalar colour map for values
                norm = colors.Normalize(ys[:,i].min(), ys[:,i].max())
                scalar_map = cm.ScalarMappable(norm=norm, cmap=self.cmap)

                # plot points between max and min and color by value
                plt.scatter(embedding[:,0], embedding[:,1],
                            c=ys[:,i],
                            vmin=ys[:,i].min(),
                            vmax=ys[:,i].max(),
                            marker='.',
                            cmap=self.cmap)
                # create a colorbar
                plt.colorbar(scalar_map)

            elif target_attr in 'habitat':

                # decode the categorical truth label
                labels = np.array([decoder[target_attr][int(y)] for y in ys[:, i]], dtype='<U3')

                # collect the unique labels
                unique_labels = np.unique(labels)
                unique_labels.sort()

                # create a colormap
                colour_key = cm.get_cmap(self.cmap)(np.linspace(0, 1, len(unique_labels)))

                label_colour_mapping = {k: colors.to_hex(colour_key[i])
                                        for i, k in enumerate(unique_labels)}

                colours = [str(label_colour_mapping[label])
                           for i, label in enumerate(labels)]

                # legend labels
                legend_elements = [Patch(facecolor=colour_key[i], label=unique_labels[i])
                                   for i, _ in enumerate(unique_labels)]

                # plot points
                plt.scatter(embedding[:, 0], embedding[:, 1],
                            marker='.',
                            label=labels,
                            c=colours)  # type: ignore

                # sort legend
                plt.legend(handles=legend_elements,
                          bbox_to_anchor=(0, 1),
                          loc="upper right")

                # remove all axis ticks on main plot
                plt.xticks([])
                plt.yticks([])


            if split:
                fig.suptitle(f"{self.model_name}\n"
                             f"UMAP on latent representation z\n"
                             f"Target Label: {target_attr}\n"
                             f"Split: {split}\n", ha='center')
            else:
                fig.suptitle(f"{self.model_name}\n"
                             f"UMAP on latent representation z\n"
                             f"Target Label: {target_attr}", ha='center')
            if self.save_path:
                print('Saving embeddings')
                df = pd.DataFrame({"s": ss,
                    "x": embedding[:, 0],
                    "y": embedding[:, 1],
                    "label": ys[:,i]})
                if split:
                    filename = f"{self.model_name}_{target_attr}_{split}.csv"
                else:
                    filename = f"{self.model_name}_{target_attr}.csv"
                df.to_csv(f"{self.save_path}/{filename}")

            yield fig
