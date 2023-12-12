from typing import List, Tuple, Any, Union, Any, Dict, Optional, Sequence

from torch.utils.data import Sampler
from torch import nn

from conduit.data.datasets.audio.ecoacoustics import  SoundscapeAttr
from conduit.data.datamodules.audio.ecoacoustics import EcoacousticsDataModule
from conduit.data.structures import TernarySample, TrainValTestSplit
from conduit.data.datasets.utils import AudioTform, CdtDataLoader
from conduit.data.datasets.wrappers import AudioTransformer

from numpy import delete, where, isin, random, isnan, concatenate, array

from pytorch_lightning.utilities.combined_loader import CombinedLoader

from typing_extensions import override
import attrs
import attr

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.transforms.transforms import LogMelSpectrogram, Framing
from src.data.ecoacoustics import Ecoacoustics
from src.data.ecoacoustics import MDAttr


@attrs.define()
class EcoacousticsDataModuleExt(EcoacousticsDataModule):

    # define new parameters for the Ecoacoustice Datamodule
    specgram_params: Dict[str, float] = attrs.field(alias="specgram_params", default=None)
    md_attrs: List[MDAttr] = list(MDAttr) 
    apply_split: bool = attrs.field(alias="apply_split", default=False) # create an equal dataset

    def prepare_data(self,
                     *args: Any,
                     **kwargs: Any) -> None:

        # download the data
        data = Ecoacoustics(
            root=self.root,
            download=True,
            segment_len=self.segment_len,
            target_attrs=self.target_attrs,
            md_attrs=self.md_attrs
        )

    def train_dataloader(self, **kwargs) -> CdtDataLoader:
        train_loader = self.make_dataloader(ds=self.train_data,
                                            batch_size=self.train_batch_size,
                                            shuffle=True,
                                            **kwargs)
        return train_loader

    def val_dataloader(self, **kwargs) -> CdtDataLoader:

        val_loader = self.make_dataloader(ds=self.val_data,
                                            batch_size=self.eval_batch_size,
                                            shuffle=False,
                                            **kwargs)
        return val_loader

    def test_dataloader(self, **kwargs) -> CdtDataLoader:
        test_loader = self.make_dataloader(ds=self.test_data,
                                            batch_size=self.eval_batch_size,
                                            shuffle=False,
                                            **kwargs)
        return test_loader


    def predict_dataloader(self, **kwargs) -> CdtDataLoader:
        val_loader = self.make_dataloader(ds=self.val_data,
                                                   batch_size=self.eval_batch_size,
                                                   **kwargs)
        train_loader = self.make_dataloader(ds=self.train_data,
                                            batch_size=self.train_batch_size,
                                            **kwargs)
        # group together into combined loader
        loaders = {'train': train_loader, 'val': val_loader }
        return CombinedLoader(loaders, mode="sequential")

    @staticmethod
    @override
    def _batch_converter(batch: TernarySample) -> TernarySample:
        return TernarySample(x=batch.x, y=batch.y, s=batch.s)

    @override
    def make_dataloader(
        self,
        ds: AudioTransformer,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        ) -> CdtDataLoader[TernarySample]:
        """Make DataLoader."""
        return CdtDataLoader(
            ds,
            batch_size=batch_size if batch_sampler is None else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
            converter=self._batch_converter,
        )

    @property
    @override
    def _default_transform(self) -> nn.Sequential:
        logmel = LogMelSpectrogram(**self.specgram_params)
        return nn.Sequential(logmel, Framing())

    @override
    def _get_audio_splits(self) -> TrainValTestSplit[Ecoacoustics]:
        # load the dataset
        "Override to refine data being used in data splits"
        all_data = Ecoacoustics(
            root=self.root,
            transform=None,
            segment_len=self.segment_len,
            target_attrs=self.target_attrs,
            md_attrs=self.md_attrs,
            sample_rate=self.sample_rate,
            download=False,
        )

        if self.apply_split:
            # create data subsets by dropping Ecuador data points
            return TrainValTestSplit(**self.drop_data(all_data, props={"val": self.val_prop, "test": self.test_prop}, seed=self.seed))
        else:
            val_data, test_data, train_data = all_data.random_split(
                props=(self.val_prop, self.test_prop), seed=self.seed
            )
            return TrainValTestSplit(train=train_data, val=val_data, test=test_data)

    def drop_data(self,
                  data: Ecoacoustics,
                  props: Dict[str, float],
                  seed: int) -> Dict[str, Ecoacoustics]:

        "Creates a set of indices that contains an even amount of data points from both countries"

        random.seed(seed)

        props["train"] = 1 - sum(props.values())  # type: ignore
        percentage_to_drop = 0.25 # Specify the percentage of values to drop
        indexes : List[int] = [] # Refined data indices
        habitats = data.metadata["habitat"].unique().tolist() # All habitats

        # go through every habitat
        for habitat in habitats:
            # get all indexes in meta csv for the habitat
            habitat_samples = data.metadata.query(f"habitat == @habitat").index.to_numpy().ravel() #tolist()

            # drop 25% of data in all EC habitats
            if 'EC' in habitat:
                # Calculate the number of values to drop based on the percentage
                num_values_to_drop = int(len(habitat_samples) * percentage_to_drop)

                # Randomly drop the specified percentage of values
                indices_to_drop = random.choice(len(habitat_samples), size=num_values_to_drop, replace=False)
                habitat_idx = delete(habitat_samples, indices_to_drop)

                # add the data points after removal of 25%
                indexes.extend(habitat_idx.tolist())
            else:
                # all uk datapoints are kept
                indexes.extend(habitat_samples.tolist())

        indexes = array(indexes)
        splits = {}
        ind_len = len(indexes)

        for split, proportion in props.items():

            # Calculate the number of values to select based on the proportion of original data
            num_values = int(ind_len * proportion)

            # Randomly select the specified proportion of values without replacement
            values = random.choice(len(indexes), size=num_values, replace=False)

            # Append the selected values to the splits dict
            splits.update({split:indexes[values].tolist()})

            # Remove selected values from the array to prevent repetition
            indexes = delete(indexes, values)  # where(isin(indexes, values)))

        return {split_name: data.subset(split_idx) for split_name, split_idx in splits.items()}


if __name__ == "__main__":
    _ = EcoacousticsDataModuleEdit()
