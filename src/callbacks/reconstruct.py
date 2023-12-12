from typing import List, Dict, Any, Iterator

import wandb
from torch import Tensor
import pytorch_lightning as pl

from matplotlib.figure import Figure
from conduit.data import TernarySample

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.transforms.transforms import Framing
from src.utils.utils import generate_title_string

class Reconstructor(pl.Callback):
    def __init__(self,
                 frame: Framing,
                 specgram_params: Dict[str, Any],
                 reconstruct_step: int = 40,
                 perc_recon: float = 0.25
                 ) -> None:
        super().__init__()
        '''
        A callback to reconstruct the output of the VAE
        '''

        self.reconstruct_step = reconstruct_step # how often we recreate training reconstructions
        self.specgram_params = specgram_params # required for librosa to plot spectrogram
        self.backward = frame.backward # required in deframing the ouputs of the model
        self.perc_recon = perc_recon # amount of reconstructions to produce per batch


    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: Dict[str, Tensor],
                           batch: TernarySample,
                           batch_idx: int) -> None:

        # recreate every 40 global steps
        if trainer.global_step % self.reconstruct_step == 0:
            # this import here stopped the psutil error, and tkinker error of image not bein gin main thread
            from matplotlib import pyplot as plt
            for fig in self.generate_figures(trainer, outputs):
                pl_module.logger.experiment.log({"train/spectrograms": wandb.Image(fig) })
                plt.close(fig)

    def on_validation_batch_end(self,
                                trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: TernarySample,
                                batch_idx: int,
                                ) -> None:
        from matplotlib import pyplot as plt
        for fig in self.generate_figures(trainer ,outputs):
            pl_module.logger.experiment.log({"val/spectrograms": wandb.Image(fig) })
            plt.close(fig)

    def on_test_batch_end(self,
                                trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: Dict[str, Tensor],
                                batch: TernarySample,
                                batch_idx: int,
                                ) -> None:

        from matplotlib import pyplot as plt
        for fig in self.generate_figures(trainer ,outputs):
            pl_module.logger.experiment.log({"test/spectrograms": wandb.Image(fig) })
            plt.close(fig)


    def generate_figures(self,
                   trainer: pl.Trainer,
                   outputs: Dict[str, Tensor]) -> Iterator[Figure]:

        # loading  modules here stops internel rm tree os error and tkinter main thread error
        import librosa
        import numpy as np
        from librosa import display as libd
        from matplotlib import pyplot as plt
        from matplotlib import gridspec as gs

        # extract outputs required
        xs, x_hats, ys, x_stds, ss= [outputs[key] for key in ["x", "x_hat", "y", "xlogvar", "s"]]

        x_stds = (0.5 * outputs["xlogvar"]).exp()
        num_rows = 3
        mask = np.random.randint(0, xs.size(0), int(xs.size(0)*self.perc_recon)) # take a percentage of the batch

        # create and yield a figure for each observation
        for j, (x, x_hat, y, x_std, s) in enumerate(zip(xs[mask], x_hats[mask], ys[mask], x_stds[mask], ss[mask])):
            # create a figure and grid spec
            fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
            grid_spec = gs.GridSpec(num_rows, 2,
                                    width_ratios=[1, 0.04],
                                    hspace=0.8)

            # deframe
            x = self.backward(x).cpu().exp()
            x_hat = self.backward(x_hat).cpu().exp()
            x_std = self.backward(x_std).cpu().exp()

            # convert to db range
            x = librosa.amplitude_to_db(x.numpy())
            x_hat = librosa.amplitude_to_db(x_hat.numpy())
            x_std = librosa.amplitude_to_db(x_std.numpy())

            # get min/max values for colourbar boundaries
            v_min = min(x.min(), x_hat.min())
            v_max = max(x.max(), x_hat.max())

            # plot observation, reconstruction and uncertainty spectrograms
            ax1 = fig.add_subplot(grid_spec[0, 0])
            ax2 = fig.add_subplot(grid_spec[1, 0])
            ax3 = fig.add_subplot(grid_spec[2, 0])

            # process original input
            mesh_1 = libd.specshow(x.squeeze(0),
                                   vmin=v_min,
                                   vmax=v_max,
                                   ax=ax1,
                                   **self.specgram_params)

            plt.colorbar(mesh_1, 
                         format='%+3.1f dB',
                         cax=fig.add_subplot(grid_spec[0, 1]),
                         orientation="vertical")

            ax1.set_title("Input Mel Spectrogram", fontsize='medium')

            # process reconstruction
            recon_spec = libd.specshow(x_hat.squeeze(0),
                                       vmin=v_min,
                                       vmax=v_max,
                                       ax=ax2,
                                       **self.specgram_params)
            # add a colourbar
            plt.colorbar(recon_spec, 
                         format='%+3.1f dB',
                         cax=fig.add_subplot(grid_spec[1, 1]),
                         orientation="vertical")

            ax2.set_title("Reconstructed Mel Spectrogram", fontsize='medium')

            # splot reconstruction
            mesh_3 = libd.specshow(x_std.squeeze(0),
                                   ax=ax3,
                                   **self.specgram_params
                                   )
            # add a colourbar
            plt.colorbar(mesh_3,
                         format='%+3.1f dB',
                         cax=fig.add_subplot(grid_spec[2, 1]),
                         orientation="vertical")

            ax3.set_title("Uncertainty Spectrogram", fontsize='medium')

            # set a title
            suptitle = generate_title_string(
                    trainer.datamodule.train_data.dataset.decoder,
                    trainer.datamodule.target_attrs,
                    y,
                    int(s)
                    )

            fig.suptitle(suptitle, wrap=True)

            yield fig
