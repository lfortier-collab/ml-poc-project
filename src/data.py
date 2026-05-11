
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ── Constantes ────────────────────────────────────────────────────────────────
PASS_THRESHOLD = 10
CAP_ABSENCES   = 30
BINARY_YES_NO  = ['schoolsup', 'famsup', 'paid', 'activities',
                   'nursery', 'higher', 'internet', 'romantic']
NOMINAL_COLS   = ['Mjob', 'Fjob', 'reason', 'guardian', 'course']


# ── Pipeline interne ───────────────────────────────────────────────────────────

def _load_raw() -> pd.DataFrame:
    df_mat = pd.read_csv(_DATA_DIR / 'student-mat.csv', sep=';')
    df_por = pd.read_csv(_DATA_DIR / 'student-por.csv', sep=';')
    df_mat['course'] = 'math'
    df_por['course'] = 'portuguese'
    df = pd.concat([df_mat, df_por], ignore_index=True)
    df['pass'] = (df['G3'] >= PASS_THRESHOLD).astype(int)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['absences'] = df['absences'].clip(upper=CAP_ABSENCES)
    df = df.drop(columns=['G1', 'G2', 'G3'])
    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['alc_total']     = df['Dalc'] + df['Walc']
    df['alc_high_risk'] = (df['alc_total'] >= 5).astype(int)
    df['parent_edu']    = (df['Medu'] + df['Fedu']) / 2
    df = df.drop(columns=['Dalc', 'Walc', 'Medu', 'Fedu'])

    df['study_vs_social']          = df['studytime'] - df['goout']
    df['motivated_with_resources'] = ((df['higher'] == 'yes') & (df['internet'] == 'yes')).astype(int)
    df['family_capital']           = df['parent_edu'] * df['famrel']
    df['has_support']              = ((df['schoolsup'] == 'yes') | (df['famsup'] == 'yes')).astype(int)
    df['digital_access']           = ((df['address'] == 'U') & (df['internet'] == 'yes')).astype(int)
    df['risk_score']               = (
        df['failures'] * 2
        + df['alc_high_risk']
        + (df['absences'] > 10).astype(int)
        + (df['studytime'] == 1).astype(int)
        - (df['higher'] == 'yes').astype(int)
    ).clip(lower=0)
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_YES_NO:
        df[col] = (df[col] == 'yes').astype(int)
    df['sex']     = (df['sex'] == 'F').astype(int)
    df['address'] = (df['address'] == 'U').astype(int)
    df['famsize'] = (df['famsize'] == 'GT3').astype(int)
    df['Pstatus'] = (df['Pstatus'] == 'T').astype(int)
    df['school']  = (df['school'] == 'GP').astype(int)
    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=False, dtype=int)
    return df


# ── Contrat imposé ─────────────────────────────────────────────────────────────

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Typical choices for the return types are ``pandas.DataFrame`` /
    ``pandas.Series`` or ``numpy.ndarray``.
    """
    df = _load_raw()
    df = _clean(df)
    df = _engineer(df)
    df = _encode(df)

    X = df.drop(columns=['pass'])
    y = df['pass']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    return X_train, X_test, y_train, y_test