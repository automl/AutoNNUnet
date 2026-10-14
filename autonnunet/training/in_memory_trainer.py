"""AutoNNUNetTrainer variant that caches the whole preprocessed dataset in RAM."""
from __future__ import annotations

from autonnunet.training.auto_nnunet_trainer import AutoNNUNetTrainer
from autonnunet.training.in_memory_dataset import InMemoryNNUNetDataset


class InMemoryAutoNNUNetTrainer(AutoNNUNetTrainer):
    """AutoNNUNetTrainer that builds its train/val datasets as
    `InMemoryNNUNetDataset` instead of the default `nnUNetDataset`, so all
    case data is loaded into RAM once at startup instead of being re-read
    from disk on every batch/epoch.

    Everything else (architecture, loss, optimizer, checkpointing,
    `from_config`) is inherited unchanged from `AutoNNUNetTrainer`.
    """

    def get_tr_and_val_datasets(self) -> tuple[InMemoryNNUNetDataset, InMemoryNNUNetDataset]:
        """Same split/construction logic as `nnUNetTrainer.get_tr_and_val_datasets`,
        just backed by `InMemoryNNUNetDataset`.
        """
        tr_keys, val_keys = self.do_split()

        dataset_tr = InMemoryNNUNetDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        dataset_val = InMemoryNNUNetDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
        )
        return dataset_tr, dataset_val
