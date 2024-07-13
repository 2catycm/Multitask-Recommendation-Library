import numpy as np
import pandas as pd
from datasets.abstract_dataset import MultitaskDataset


class Anime(MultitaskDataset):
    
    def __init__(self, dataset_path):
        super().__init__()
        processed_anime = pd.read_csv(dataset_path + "processed_anime.csv")
        rating = pd.read_csv(dataset_path + "rating.csv")
        data = (pd.merge(processed_anime, rating, how="inner", on="anime_id")).to_numpy()


        super().set_param(
            numerical_data=data[:, :1].astype(np.float32),
            categorical_data=data[:, 1:-1].astype(np.int32),
            labels=data[:, -1:].astype(np.int32)
        )
