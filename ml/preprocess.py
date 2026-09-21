import numpy as np
import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует целевую переменную Target.
    """

    df = df.copy()

    df["Target"] = np.where(
        df["MaxOverdueDays90"] >= 25,
        1,
        0,
    )

    df = df.drop(columns=["MaxOverdueDays90"])

    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(
        columns=["ApplicationId", "ec.ApplicationId", "BirthDate", "ApplicationDate"],
        errors="ignore",
    )

    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns

    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df[num_cols] = df[num_cols].fillna(0)

    return df
