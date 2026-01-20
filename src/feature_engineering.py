import pandas as pd
def prepare_intent_features(csv_path):
    df = pd.read_csv(csv_path)

    X = df[["disaster", "action", "urgency"]]
    y = df["label"]

    return X, y
def prepare_verification_features(csv_path):
    df = pd.read_csv(csv_path)

    X = df[[
        "trust",
        "gov_source",
        "panic",
        "corroboration"
    ]]

    y = df["label"]

    return X, y
