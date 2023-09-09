import numpy as np
import pandas as pd
from pandas import DataFrame
from pynndescent import NNDescent

from dataset.regression import get_scores


def make_vector(data: DataFrame, video_ids: list[int]):
    data = data[data["item_id"].isin(video_ids)]

    #TODO add weights

    return np.mean(data.drop(["item_id"], axis=1).values)


def get_video_corpus(data: DataFrame, index: NNDescent, vector: np.ndarray):
    top100nearest = index.query(vector, k=100)
    vids_ids = top100nearest[0]
    return data[data["item_id"].isin(vids_ids)].values, vids_ids


def get_popular_videos():
    pass


def create_submission_file(path: str, data: DataFrame, index: NNDescent):
    test_file = pd.read_csv(path)
    user_ids = test_file["user_id"].values
    preds = []
    for user_id in user_ids:
        x_predict = make_vector(data, test_file[test_file["user_id"] == user_id]["video_id"].values)
        if len(x_predict) == 0:
            preds.append(get_popular_videos())
            break
        corpus, target = get_video_corpus(data, index, x_predict)
        preds.append(get_scores(corpus, target, x_predict, return_names=True, return_only_names=True))

    submission = pd.DataFrame({"user_id": user_ids, "recs": preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    index = NNDescent(np.load("vectors.npy"), n_jobs=16)
    data = pd.read_csv("data.csv")
    create_submission_file("test.csv", data, index)
