import json

from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomTreesEmbedding, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from tqdm import tqdm

from utils import transform_data, pack_arguments, test_model

train_df = pd.read_csv("datasets/train.csv")
#train_df = train_df.drop(labels="Name", axis=1)

"""
#test_df = pd.read_csv("datasets/test.csv")
#test_df = test_df.drop(labels="Name", axis=1)
categorical_features = ["HomePlanet", "Cabin", "Destination"]


target_col = "Transported"
feature_cols = [
    "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
]
for col in feature_cols:
    if col in categorical_features:
        train_df[col] = train_df[col].astype("category")
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    else:
        train_df[col] = train_df[col].fillna(train_df[col].median())

cabin_data = list(zip(*map(lambda s: s.split("/"), train_df["Cabin"].tolist())))
train_df["Cabin_deck"] = cabin_data[0]
train_df["Cabin_num"] = cabin_data[1]
train_df["Cabin_side"] = cabin_data[2]
train_df = train_df.drop(labels="Cabin", axis=1)
categorical_features = ["HomePlanet", "Cabin_deck", "Cabin_num", "Cabin_side", "Destination"]
numerical_features = [
    "CryoSleep", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
]
"""
#WITH ONE HOT CAT: XGBClassifier/eval_metric=logloss/colsample_bytree=1.0/subsample=1.0/max_depth=7/learning_rate=0.1/n_estimators=50 0.8067855089131685
#WITH ORDINAL CATEGORY: GradientBoostingClassifier/min_samples_split=5/subsample=1.0/max_depth=7/learning_rate=0.05/n_estimators=100 0.8055606925591278


feature_cols = [
    "HomePlanet", "CryoSleep", "Cabin_deck", "Cabin_num", "Cabin_side", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
]
categorical_features = ["HomePlanet", "Cabin_deck", "Cabin_side", "Destination", "LastName", "Group", "CryoSleep", "VIP"]
numerical_features = ["Cabin_num", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpend"]

target_col = "Transported"
train_df = transform_data(train_df)

column_transformer = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),

        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numerical_features),
    ],
    remainder="drop"
)


X = train_df.drop(columns=target_col, axis=1)
Y = train_df[target_col]

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
random_states = [42, 4242, 2026, 2008, 1914, 1984]
data = {seed: train_test_split(X, Y, test_size=0.2, random_state=seed) for seed in random_states}

test_list = [
    (DecisionTreeClassifier, {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "ccp_alpha": [0.0, 0.01, 0.1]
    }),

    (ExtraTreeClassifier, {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }),

    (GradientBoostingClassifier, {
        "n_estimators": [100],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5]
    }),

    (XGBClassifier, {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "eval_metric": ["logloss"]
    }),

    (XGBRFClassifier, {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.8],
        "colsample_bynode": [0.8],
        "eval_metric": ["logloss"]
    })
]
r = []

best = None
best_score = -1
for i in range(len(test_list)):
    model_cls, args = test_list[i]
    model_cls_name = model_cls.__qualname__

    packed_args = pack_arguments(model_cls_name, args)

    print(f"TESTIBG {model_cls_name}")
    for args in tqdm(packed_args):
        scores = []
        for random_state in random_states:
            x_train, x_test, y_train, y_test = data[random_state]
            name = args["name"]

            model_args = {
                key: value
                for key, value in args.items()
                if key != "name"
            }

            model = model_cls(**model_args, random_state=random_state)
            pipeline = Pipeline([
                ("preprocessor", column_transformer),
                ("model", model)
            ])

            score = test_model(pipeline, x_train, x_test, y_train, y_test)
            scores.append(score)

        scores = np.array(scores)
        stat_score = scores.mean() - scores.std()

        if stat_score > best_score:
            best_score = stat_score
            best = name

        r.append((name, stat_score))
print(r)
print(best, best_score)
with open("models/models.json", "w+") as f:
    f.write(json.dumps(r))
#linear_preprocessor = ColumnTransformer(
#    transformers=[
#        ("cat", OneHotEncoder(), categorical_features),
#        ("num", StandardScaler(), numerical_features)
#    ]
#)
#linear_train_x = linear_preprocessor.fit_transform(train_x)