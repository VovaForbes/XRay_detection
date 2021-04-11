import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def train_val_split_folds(df, seed, n_folds):
    indices_df = pd.DataFrame(df["image_id"].unique(), columns=list(["Index"]))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_val_splits_generator = kf.split(indices_df)

    train_val_splits = []

    for train_idx, val_idx in train_val_splits_generator:
        train_ids = indices_df.iloc[train_idx].Index.values
        val_ids = indices_df.iloc[val_idx].Index.values
        train_df = df.loc[df["image_id"].isin(train_ids)]
        val_df = df.loc[df["image_id"].isin(val_ids)]
        train_val_splits.append((train_df, val_df))

        # Make sure there is no intersection between val and train in case of several bboxes
        assert set(train_df["image_id"].unique()).intersection(set(val_df["image_id"].unique())) == set()

    return train_val_splits
