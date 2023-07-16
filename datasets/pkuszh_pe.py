import numpy as np
import pandas as pd
from datasets.abstract_dataset import MultitaskDataset


class PkuszhPE(MultitaskDataset):
    """北大深圳医院体检数据集
    """
    def __init__(self, dataset_path):
        super().__init__()
        data = pd.read_csv(dataset_path).to_numpy()

        super().set_param(
            numerical_data=data[:, :42].astype(np.float32),
            categorical_data=data[:, 42:-7].astype(np.int32),
            labels=data[:, -7:].astype(np.float32)
        )
