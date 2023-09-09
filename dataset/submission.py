import numpy as np
import pandas as pd
from pandas import DataFrame
from pynndescent import NNDescent
from sklearn import preprocessing
from tqdm import tqdm

from dataset.regression import get_scores


def vid_long(videos: DataFrame, user_id: int, users: DataFrame) -> DataFrame:
    tqdm.pandas()
    hist = users[users["user_id"] == user_id]
    hist = pd.merge(hist, videos, on='item_id').drop(columns=["user_id"], axis=1)
    for i in tqdm(range(100)):
        hist[f"v_title_{i}"] = hist.progress_apply(lambda row: row[f"v_title_{i}"] * 2
        if ((row["watch_time"] / (row["duration"] / 1000)) > 0.25
            if (row["duration"] / 1000) > 300
            else row["watch_time"] > 30)
        else 1, axis=1)
        hist[f"v_title_{i}"] = preprocessing.minmax_scale(hist[f"v_title_{i}"].T).T
    return hist.drop(columns=["watch_time", "duration"], axis=1)


def like(videos: pd.DataFrame, user_id: int, users: pd.DataFrame, emotions: pd.DataFrame) -> pd.DataFrame:
    emotions = emotions[["C2", "C3", "C4"]]
    emotions = emotions[emotions["C2"] == user_id]
    emotions = emotions[["C3", "C4"]]
    hist = users[users["user_id"] == user_id]
    hist = pd.merge(hist, videos, on='item_id').drop(columns=["user_id"], axis=1)
    hist = pd.merge(hist, emotions, left_on='item_id', right_on="C3").drop(columns=["user_id"], axis=1)
    for i in tqdm(range(100)):
        hist[f"v_title_{i}"] = hist.progress_apply(lambda row: row[f"v_title_{i}"] * 2 if row["C4"] == "pos_emotions"
        else (0.5 if row["C4"] == "neg_emotions" else 1), axis=1)
        hist[f"v_title_{i}"] = preprocessing.minmax_scale(hist[f"v_title_{i}"].T).T
    return hist.drop(columns=["C4", "C3"], axis=1)


def make_vector(data: DataFrame, video_ids: list[int], users: DataFrame, emotions: DataFrame):
    data = data[data["item_id"].isin(video_ids)]

    vid_long(data, 1, users)
    like(data, 1, users, emotions)
    # TODO add weights

    return np.mean(data.drop(["item_id"], axis=1).values)


def get_video_corpus(data: DataFrame, index: NNDescent, vector: np.ndarray):
    top100nearest = index.query(vector, k=100)
    vids_ids = top100nearest[0]
    return data[data["item_id"].isin(vids_ids)].values, vids_ids


def get_10_category(train_hist: pd.DataFrame, new_hist: pd.DataFrame = None) -> list:
    if new_hist is not None:
        train_hist = train_hist.append(new_hist, ignore_index=True)
    cat_columns = [col for col in tqdm(train_hist.columns) if col.startswith('cat')]
    cat_data = train_hist[cat_columns]

    # подсчет количества 1 в каждом столбце
    counts = cat_data.sum()

    # сортировка столбцов по убыванию количества 1 и вывод первых 10
    top_columns = counts.sort_values(ascending=False)[:10]
    return top_columns


def get_top_videos_in_cat(cat_name, train_hist: pd.DataFrame, new_hist: pd.DataFrame = None) -> str:
    if new_hist is not None:
        train_hist = train_hist.append(new_hist, ignore_index=True)
    cat_data = train_hist.loc[train_hist[cat_name] == 1]
    counts = cat_data['item_id'].value_counts()
    return counts.index[0]


def get_top_videos(train_hist: pd.DataFrame, new_hist: pd.DataFrame = None) -> list:
    top_category = get_10_category(train_hist, new_hist)
    return [get_top_videos_in_cat(i, train_hist, new_hist) for i in tqdm(top_category)]


def create_submission_file(path: str, data: DataFrame, index: NNDescent, users: DataFrame, emotions: DataFrame,
                           train_hist: DataFrame, new_hist: DataFrame):
    test_file = pd.read_csv(path)
    user_ids = test_file["user_id"].values
    preds = []
    for user_id in user_ids:
        x_predict = make_vector(data, test_file[test_file["user_id"] == user_id]["video_id"].values, users, emotions)
        if len(x_predict) == 0:
            get_top_videos(train_hist, new_hist)
            break
        corpus, target = get_video_corpus(data, index, x_predict)
        preds.append(get_scores(corpus, target, x_predict, return_names=True, return_only_names=True))

    submission = pd.DataFrame({"user_id": user_ids, "recs": preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    index = NNDescent(np.load("vectors.npy"), n_jobs=16)
    index.prepare()
    data = pd.read_csv("data.csv")
    users = pd.read_parquet('player_starts_train.parquet', engine='pyarrow')
    # users = pd.read_csv("player_starts_train.csv")
    train_hist = pd.merge(users, data, on='item_id').drop(columns=["user_id"], axis=1)
    try:
        new_users = pd.read_csv("new_player_starts_train.csv")
    except FileNotFoundError:
        new_users = None
    new_hist = pd.merge(new_users, data, on='item_id').drop(columns=["user_id"], axis=1)
    emotions = pd.read_csv("emotions.csv")
    create_submission_file("test.csv", data, index, users, emotions, train_hist, new_hist)
