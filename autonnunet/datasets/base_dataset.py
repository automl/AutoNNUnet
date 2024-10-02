from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from autonnunet.utils.paths import (NNUNET_DATASETS, NNUNET_PREPROCESSED,
                                    NNUNET_RAW)


class Dataset(ABC):
    def __init__(self, name: str):
        self.name = name
        self.dataset_id = int(name.split("_")[0].replace("Dataset", ""))

        self.dl_dataset_dir = NNUNET_DATASETS

        self.raw_dataset_dir = NNUNET_RAW
        self.raw_dataset_path = self.raw_dataset_dir / self.name

        self.preprocessed_dataset_dir = NNUNET_PREPROCESSED
        self.preprocessed_dataset_path = self.preprocessed_dataset_dir / self.name

        self.logger = logging.getLogger("Dataset")

    @abstractmethod
    def download_and_extract(self):
        pass

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass
