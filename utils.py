from pandas import DataFrame
import pandas as pd

def transform_data(frame: DataFrame):
    frame = frame.drop(labels="Name", axis=1)
    frame = frame.drop(labels="PassengerId", axis=1)

    categorical_features = ["HomePlanet", "Cabin", "Destination"]
    feature_cols = [
        "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
    ]
    for col in feature_cols:
        if col in categorical_features:
            frame[col] = frame[col].astype("category")
            frame[col] = frame[col].fillna(frame[col].mode()[0])
        else:
            frame[col] = frame[col].fillna(frame[col].median())

    frame["VIP"] = frame["VIP"].astype(bool)
    frame["CryoSleep"] = frame["CryoSleep"].astype(bool)

    cabin_data = list(zip(*map(lambda s: s.split("/"), frame["Cabin"].tolist())))
    frame["Cabin_deck"] = cabin_data[0]
    frame["Cabin_num"] = pd.to_numeric(cabin_data[1])
    frame["Cabin_side"] = cabin_data[2]
    #frame = pd.get_dummies(frame, columns=["HomePlanet", "Cabin_deck", "Cabin_side", "Destination"])
    frame = frame.drop(labels="Cabin", axis=1)

    return frame


def test_model(pipeline, x_train, x_test, y_train, y_test):
    pipeline.fit(x_train, y_train)

    return pipeline.score(x_test, y_test)


def pack_arguments(algo_name: str, kwargs: dict):
    if len(kwargs) == 0:
        return {"name": algo_name}

    key = list(kwargs.keys())[0]
    if len(kwargs) == 1:
        return [{key: v, "name": f"{algo_name}/{key}={v}"} for v in kwargs[key]]

    r = []
    cur_values = kwargs.pop(key)
    for resulted_args in pack_arguments(algo_name, kwargs.copy()):
        for v in cur_values:
            newd = {key: v, **resulted_args}
            newd["name"] += f"/{key}={v}"
            r.append(newd)

    return r
