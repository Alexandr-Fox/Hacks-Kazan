import pandas as pd
from sklearn import neighbors

IMPORT_FILE = "small_player_starts_train.csv"
DATA_FILE = "new_data.csv"
N = 10

def create_data_file():
    import_data = pd.read_csv(f"dataset/{IMPORT_FILE}", sep=',')
    auth_data = import_data[import_data.is_autorized == 1]
    auth_data.reset_index(drop=True, inplace=True)
    auth_data.drop(['is_autorized', 'date'], axis= 1 , inplace=True)
    auth_data.rename(columns = {'item_id':'video_id'}, inplace=True)
    auth_data['user_id'] = auth_data['user_id'].apply(lambda x: x[5:])
    auth_data['video_id'] = auth_data['video_id'].apply(lambda x: x[6:])

    #print(auth_data)
    auth_data.to_csv(f"dataset/{DATA_FILE}", sep=',', index=False)

def main(user, video):
    data = pd.read_csv(f"dataset/{DATA_FILE}", sep=',')
    #print(data)

    x_data = data[['user_id', 'video_id']].copy()
    y_data = data[['watch_time']].copy()

    knn = neighbors.KNeighborsClassifier(n_neighbors = N)
    knn.fit(x_data, y_data)
    test_dict = {"user_id":[user], "video_id":[video]}
    test_data = pd.DataFrame(test_dict)
    ans = knn.predict(test_data)
    print()
    print(ans)
    return 0

if __name__ == "__main__":
    create_data_file()
    main(user=9427665, video=320208)