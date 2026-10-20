import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
CAT_COLS = ["C3_Complaint_Type", "department", "suburb_group", "Ward", "Sub_Council"]
FEATURES = CAT_COLS + ["Created_On_Date"]

BANNED = {"Completed_Date", "Changed_on", "Notification_type", "Notifications_Created",
          "resolution_days", "target", "Suburb", "Work_Center", "Notification"}
assert not BANNED & set(FEATURES), "Leaky or sensitive column in features!"


def time_features(X):
    X = X.copy()
    t = pd.to_datetime(X["Created_On_Date"])
    X["month"] = t.dt.month.astype(str)
    X["weekday"] = t.dt.weekday.astype(str)
    X["is_weekend"] = (t.dt.weekday >= 5).astype(int)
    for c in CAT_COLS:
        s = X[c].astype("object").where(X[c].notna(), "MISSING").astype(str).str.strip()
        X[c] = s.where(~s.isin(["", "#", "nan", "<NA>"]), "MISSING")
    return X.drop(columns=["Created_On_Date"])


def build_preprocessor(drop=(), min_freq=100):
    cats = [c for c in CAT_COLS + ["month", "weekday"] if c not in drop]
    nums = [c for c in ["is_weekend"] if c not in drop]
    return Pipeline([
        ("time", FunctionTransformer(time_features)),
        ("cols", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_freq, sparse_output=True), cats),
            ("num", StandardScaler(), nums),
        ])),
    ])
