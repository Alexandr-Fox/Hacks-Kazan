import pandas as pd
from dataset.regression import get_scores


def make_vector(user_id: int, video_ids: list[int]):
    pass


def get_video_corpus(user_id: int):
    pass


def create_submission_file(path: str):
    test_file = pd.read_csv(path)
    user_ids = test_file["user_id"].values
    preds = []
    for user_id in user_ids:
        x_predict = make_vector(user_id, test_file[test_file["user_id"] == user_id]["video_id"].values)
        corpus, target = get_video_corpus(x_predict)
        preds.append(get_scores(corpus, target, x_predict, return_names=True, return_only_names=True))

    submission = pd.DataFrame({"user_id": user_ids, "recs": preds})
    submission.to_csv("submission.csv", index=False)