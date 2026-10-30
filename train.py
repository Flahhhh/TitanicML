import json

from pandas import DataFrame
import pandas as pd
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
test_df = pd.read_csv("datasets/test.csv")

feature_cols = [
    "HomePlanet", "CryoSleep", "Cabin_deck", "Cabin_num", "Cabin_side", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
]
#categorical_features = ["HomePlanet", "Cabin_deck", "Cabin_side", "Destination"]
target_col = "Transported"
train_frame = transform_data(train_df)
test_frame = transform_data(test_df)



X = train_frame.drop(columns=target_col, axis=1)
Y = train_frame[target_col]
model = XGBClassifier(eval_metric="logloss", colsample_bytree=0.7, subsample=1.0, max_depth=7, learning_rate=0.1, n_estimators=100)
categorical_features = ["HomePlanet", "Cabin_deck", "Cabin_side", "Destination", "LastName", "Group", "CryoSleep", "VIP"]
numerical_features = ["Cabin_num", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpend"]
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

pipeline = Pipeline([
    ("preprocessor", column_transformer),
    ("model", model)
])

pipeline.fit(X, Y)

X_test = test_frame
result = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Transported": pipeline.predict(X_test).astype(bool)
})

result.to_csv("result.csv", index=False)
