from sklearn.linear_model import LogisticRegression
import catboost as cb


def get_scores(X, y, x_predict, return_names=False, return_only_names=False):
    if return_only_names is True and return_names is False:
        raise ValueError("return_only_names cannot be True if return_names is False")
    clf = LogisticRegression().fit(X, y)
    scores = clf.predict_proba(x_predict)
    if return_names is False:
        return scores
    else:
        classes = clf.classes_
        scores = scores.tolist()[0]
        classes = [(classes[i], scores[i]) for i in range(len(scores))]
        classes.sort(key=lambda x: x[1], reverse=True)
        if return_only_names is True:
            classes = [i[0] for i in classes]
        return classes


def get_scores_catboost(X, y, x_predict, return_names=False, return_only_names=False):
    if return_only_names is True and return_names is False:
        raise ValueError("return_only_names cannot be True if return_names is False")
    clf = cb.CatBoostClassifier(iterations=1000, learning_rate=0.01).fit(X, y)
    scores = clf.predict_proba(x_predict)
    if return_names is False:
        return scores
    else:
        classes = clf.classes_
        scores = scores.tolist()[0]
        classes = [(classes[i], scores[i]) for i in range(len(scores))]
        classes.sort(key=lambda x: x[1], reverse=True)
        if return_only_names is True:
            classes = [i[0] for i in classes]
        return classes



X, y = [[0, 0, 2, 6], [1, 1, 4, 2]], ['video_1', 'video_2']
x_predict = [[1, 2, 3, 1]]
print(get_scores_catboost(X, y, x_predict, return_names=True, return_only_names=True))
