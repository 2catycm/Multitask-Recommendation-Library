import numpy as np
import pandas as pd
from abstract_dataset import MultitaskDataset
class AliExpressDataset(MultitaskDataset):
    """
    AliExpress Dataset
    This is a dataset gathered from real-world traffic logs of the search system in AliExpress
    Reference:
        https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
        Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
    """

    def __init__(self, dataset_path):
        super().__init__()
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:]
        super().set_param(
            categorical_data=data[:, :16].astype(np.int32),
            numerical_data=data[:, 16: -2].astype(np.float32),
            labels=data[:, -2:].astype(np.float32)
        )
