import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.utils import project_path


class WatchLogDataset(Dataset):
    def __init__(self, df, scaler=None):
        self.df = df
        self.features = None
        self.labels = None
        self.contents_id_map = None
        self.scaler = scaler
        self._preprocessing()

    def _preprocessing(self):
        content_id_categories = pd.Categorical(self.df["content_id"])
        self.contents_id_map = dict(enumerate(content_id_categories.categories))
        self.df["content_id"] = content_id_categories.codes

        target_columns = ["rating", "popularity", "watch_seconds"]

        if self.scaler:
            scaled_features = self.scaler.transform(self.df[target_columns])
        else:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(self.df[target_columns])

        self.features = torch.FloatTensor(scaled_features)
        self.labels = torch.LongTensor(self.df["content_id"].values)

    def decode_content_id(self, encoded_id):
        return self.contents_id_map[encoded_id]

    @property
    def features_dim(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return len(self.df["content_id"].unique())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def read_dataset():
    watch_log_path = os.path.join(project_path(), "dataset", "watch_log.csv")
    df = pd.read_csv(watch_log_path)
    return df

def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_datasets():
    df = read_dataset()
    train_df, val_df, test_df = split_dataset(df)
    train_dataset = WatchLogDataset(train_df)
    val_dataset = WatchLogDataset(val_df)
    test_dataset = WatchLogDataset(test_df)
    return train_dataset, val_dataset, test_dataset