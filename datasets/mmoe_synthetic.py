import numpy as np
import pandas as pd
from datasets.abstract_dataset import MultitaskDataset


class SynDataset(MultitaskDataset):

    def __init__(self, dataset_path):
        super().__init__()
        data = pd.read_csv(dataset_path).to_numpy()[:, 0:]

        super().set_param(
            categorical_data=data[:, 0:0].astype(np.int32),
            numerical_data=data[:, : -2].astype(np.float32),
            labels=data[:, -2:].astype(np.float32)
        )
