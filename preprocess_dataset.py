from typing import List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils_and_constants import (
    DROP_COLNAMES,
    PROCESSED_DATASET,
    RAW_DATASET,
    TARGET_COLUMN,
)


def read_dataset(
    filename: str, drop_columns: List[str], target_column: str
) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values can be binary format (Yes/No) or numeric (0/1)

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    # 1. Lecture et suppression tolérante des colonnes optionnelles
    df = pd.read_csv(filename).drop(columns=drop_columns, errors="ignore")

    # 2. Vérification explicite de la présence de la colonne cible
    if target_column not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_column}' manquante dans le fichier {filename}. "
            f"Colonnes disponibles: {df.columns.tolist()}"
        )

    original_values = df[target_column].dropna().unique().tolist()

    # 3. Convertisseur intelligent pour accepter les 0/1 déjà présents ou les Yes/No
    def _convert_target(x):
        if pd.isna(x):
            return x
        # Si c'est déjà un nombre (0 ou 1)
        if isinstance(x, (int, float)):
            if x in (0, 1):
                return int(x)
            return None
        # Si c'est du texte (Yes, No, True, False)
        s = str(x).strip().lower()
        if s in ("yes", "y", "true", "1"):
            return 1
        if s in ("no", "n", "false", "0"):
            return 0
        return None

    df[target_column] = df[target_column].apply(_convert_target)

    # 4. Alerte si des valeurs non reconnues sont présentes
    if df[target_column].isna().any():
        raise ValueError(
            f"Conversion de la colonne cible a produit des valeurs manquantes. "
            f"Valeurs originales trouvées: {original_values}"
        )

    return df


def target_encode_categorical_features(
    df: pd.DataFrame, categorical_columns: List[str], target_column: str
) -> pd.DataFrame:
    """
    Target encodes the categorical features of the dataframe
    (http://saedsayad.com)
    """
    encoded_data = df.copy()
    for col in categorical_columns:
        encoding_map = df.groupby(col)[target_column].mean().to_dict()
        encoded_data[col] = encoded_data[col].map(encoding_map)
    return encoded_data


def impute_and_scale_data(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes numerical data to its mean value
    and then scales the data to a normal distribution
    """
    imputer = SimpleImputer(strategy="mean")
    X_preprocessed = imputer.fit_transform(df_features.values)
    scaler = StandardScaler()
    X_preprocessed = scaler.fit_transform(X_preprocessed)
    return pd.DataFrame(X_preprocessed, columns=df_features.columns)


def main():
    # Read data
    weather = read_dataset(
        filename=RAW_DATASET, drop_columns=DROP_COLNAMES, target_column=TARGET_COLUMN
    )

    # Target encode categorical columns
    categorical_columns = weather.select_dtypes(include=[object]).columns.to_list()
    weather = target_encode_categorical_features(
        df=weather, categorical_columns=categorical_columns, target_column=TARGET_COLUMN
    )

    # Impute and scale features
    weather_features_processed = impute_and_scale_data(
        weather.drop(columns=TARGET_COLUMN)
    )

    # Write processed dataset
    weather_labels = weather[TARGET_COLUMN]
    weather = pd.concat([weather_features_processed, weather_labels], axis=1)
    weather.to_csv(PROCESSED_DATASET, index=None)


if __name__ == "__main__": 
    main()
