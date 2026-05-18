"""Streamlit application — Student Success Predictor."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import shap
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import MODEL_METRICS_FILE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SklearnScaler

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS    = ROOT / "models"

COLORS = {
    "Échec":    "#E74C3C",
    "Réussite": "#2ECC71",
    "lr":       "#0275d8",
    "rf":       "#5cb85c",
    "gb":       "#d9534f",
}
sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────────────────────
# Loaders (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def _load_models() -> dict:
    train_s = pd.read_csv(DATA_PROC / "student_train.csv")
    test_s  = pd.read_csv(DATA_PROC / "student_test.csv")
    train_r = pd.read_csv(DATA_PROC / "student_train_raw.csv")
    test_r  = pd.read_csv(DATA_PROC / "student_test_raw.csv")
    X_train_r = train_r.drop(columns=["pass"])
    X_test_r  = test_r.drop(columns=["pass"])
    lr = joblib.load(MODELS / "logistic_regression.joblib")
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"

    # ── K-Means clustering sur features comportementales clés
    CLUSTER_FEATS = ["failures", "studytime", "goout", "absences",
                     "alc_total", "higher", "famrel", "parent_edu", "risk_score"]
    X_clust = X_train_r[CLUSTER_FEATS]
    clust_scaler = SklearnScaler()
    X_clust_scaled = clust_scaler.fit_transform(X_clust)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    kmeans.fit(X_clust_scaled)

    # Profiler chaque cluster pour lui donner un nom
    centers = pd.DataFrame(
        clust_scaler.inverse_transform(kmeans.cluster_centers_),
        columns=CLUSTER_FEATS,
    )
    # Avec k=2 : identifier lequel est "à risque" vs "favorable"
    # Le cluster avec le plus de failures/risk_score = profil à risque
    risk_scores = centers["risk_score"].values
    risk_idx    = int(np.argmax(risk_scores))
    safe_idx    = 1 - risk_idx
    cluster_labels = {
        risk_idx: ("🔴 Profil à risque",
                   "Échecs passés, score de risque élevé, engagement scolaire faible."),
        safe_idx: ("🟢 Profil favorable",
                   "Peu d'échecs, bon engagement scolaire, ambition d'études supérieures."),
    }

    return {
        "lr":        lr,
        "rf":        joblib.load(MODELS / "random_forest.joblib"),
        "gb":        joblib.load(MODELS / "gradient_boosting.joblib"),
        "X_train_s": train_s.drop(columns=["pass"]),
        "X_train_r": X_train_r,
        "X_test_s":  test_s.drop(columns=["pass"]),
        "X_test_r":  X_test_r,
        "y_train":   train_s["pass"],
        "y_test":    test_s["pass"],
        "features_r": list(X_train_r.columns),
        "kmeans":         kmeans,
        "clust_scaler":   clust_scaler,
        "cluster_feats":  CLUSTER_FEATS,
        "cluster_centers":centers,
        "cluster_labels": cluster_labels,
    }


@st.cache_data
def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_mat = pd.read_csv(DATA_RAW / "student-mat.csv", sep=";")
    df_por = pd.read_csv(DATA_RAW / "student-por.csv", sep=";")
    df_mat["course"] = "math"
    df_por["course"] = "portuguese"
    df = pd.concat([df_mat, df_por], ignore_index=True)
    df["pass"] = (df["G3"] >= 10).astype(int)
    return df, df_mat, df_por


# ─────────────────────────────────────────────────────────────
# Helper — compute metrics dict for one model
# ─────────────────────────────────────────────────────────────
def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "F1 macro ★":  round(f1_score(y_true, y_pred, average="macro"), 3),
        "F1 Échec":    round(f1_score(y_true, y_pred, average=None)[0], 3),
        "F1 Réussite": round(f1_score(y_true, y_pred, average=None)[1], 3),
        "Accuracy":    round(accuracy_score(y_true, y_pred), 3),
        "AUC-ROC":     round(roc_auc_score(y_true, y_proba), 3),
        "Precision":   round(precision_score(y_true, y_pred, average="macro"), 3),
        "Recall":      round(recall_score(y_true, y_pred, average="macro"), 3),
    }


@st.cache_resource
def _shap_explainer_gb(_model, _X_bg):
    explainer = shap.TreeExplainer(_model, _X_bg)
    return explainer

@st.cache_resource
def _shap_explainer_rf(_model, _X_bg):
    explainer = shap.TreeExplainer(_model, _X_bg)
    return explainer

def _best_threshold(y_true, y_proba, metric="f1_macro"):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_score = 0.5, 0.0
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred_t, average="macro")
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


# ─────────────────────────────────────────────────────────────
# Main entry point (called by scripts/main.py)
# ─────────────────────────────────────────────────────────────
def build_app() -> None:
    """Render the Student Success Predictor Streamlit application."""

    st.set_page_config(
        page_title="Student Success Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Hide default Streamlit chrome ── */
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        section[data-testid="stSidebar"] .stRadio label {
            background: rgba(255,255,255,0.06);
            border-radius: 8px; padding: .45rem .9rem;
            margin-bottom: .3rem; display: block;
            border: 1px solid rgba(255,255,255,0.08);
            transition: background .2s;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255,255,255,0.13);
        }
        section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
        section[data-testid="stSidebar"] small, section[data-testid="stSidebar"] .stCaption {
            color: #94a3b8 !important; font-size: .75rem !important;
        }

        /* ── Hero banner ── */
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2744 100%);
            border-radius: 16px; padding: 2.5rem 2.8rem;
            margin-bottom: 1.5rem; position: relative; overflow: hidden;
        }
        .hero-banner::before {
            content: ''; position: absolute; top: -50%; right: -10%;
            width: 400px; height: 400px; border-radius: 50%;
            background: radial-gradient(circle, rgba(99,179,237,0.15) 0%, transparent 70%);
        }
        .hero-title {
            font-size: 2.6rem; font-weight: 800; line-height: 1.2;
            color: #ffffff; margin: 0;
        }
        .hero-sub {
            font-size: 1.1rem; color: #94a3b8; margin-top: .6rem; margin-bottom: 0;
        }
        .hero-badge {
            display: inline-block; background: rgba(99,179,237,0.2);
            color: #63b3ed; border: 1px solid rgba(99,179,237,0.3);
            border-radius: 20px; padding: .2rem .75rem;
            font-size: .78rem; font-weight: 600; margin-bottom: .8rem;
            text-transform: uppercase; letter-spacing: .05em;
        }

        /* ── Section title ── */
        .section-title {
            font-size: 1.45rem; font-weight: 700; color: #0f172a;
            margin-top: .5rem; margin-bottom: .1rem;
            padding-bottom: .4rem;
            border-bottom: 3px solid #3b82f6;
            display: inline-block;
        }

        /* ── Cards ── */
        .card {
            background: #ffffff; border-radius: 14px;
            padding: 1.3rem 1.6rem; margin-bottom: .8rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
            border-left: 4px solid #3b82f6;
            transition: box-shadow .2s;
        }
        .card:hover { box-shadow: 0 4px 20px rgba(0,0,0,.1); }
        .card-green  { border-left-color: #10b981; }
        .card-red    { border-left-color: #ef4444; }
        .card-orange { border-left-color: #f59e0b; }

        /* ── Tags ── */
        .tag {
            display: inline-block; background: #eff6ff; color: #2563eb;
            border: 1px solid #bfdbfe; border-radius: 6px;
            padding: .15rem .6rem; font-size: .75rem;
            font-weight: 600; margin-right: .3rem; margin-bottom: .2rem;
        }
        .tag-green  { background: #f0fdf4; color: #16a34a; border-color: #bbf7d0; }
        .tag-orange { background: #fffbeb; color: #d97706; border-color: #fde68a; }
        .tag-red    { background: #fef2f2; color: #dc2626; border-color: #fecaca; }

        /* ── Metric cards ── */
        div[data-testid="metric-container"] {
            background: #f8fafc; border-radius: 12px; padding: .8rem 1rem !important;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 4px rgba(0,0,0,.05);
        }
        div[data-testid="metric-container"] label { color: #64748b !important; font-size: .82rem !important; font-weight: 500 !important; }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.6rem !important; font-weight: 700 !important; color: #0f172a !important;
        }

        /* ── Verdict box ── */
        .verdict-box {
            border-radius: 16px; padding: 1.8rem; text-align: center;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 24px rgba(0,0,0,.1);
        }
        .verdict-icon { font-size: 3rem; line-height: 1; }
        .verdict-label { font-size: 1.5rem; font-weight: 800; margin-top: .4rem; }
        .verdict-prob  { font-size: 1.1rem; margin-top: .3rem; opacity: .85; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] { gap: .5rem; border-bottom: 2px solid #e2e8f0; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0 !important; padding: .5rem 1rem !important;
            font-weight: 500 !important;
        }
        .stTabs [aria-selected="true"] { background: #eff6ff !important; color: #2563eb !important; }

        /* ── Buttons ── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
            border: none !important; border-radius: 10px !important;
            font-weight: 600 !important; font-size: 1rem !important;
            padding: .65rem 1.5rem !important;
            box-shadow: 0 4px 14px rgba(37,99,235,.3) !important;
            transition: all .2s !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(37,99,235,.45) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; }

        /* ── Info / success / error ── */
        .stAlert { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="padding:.5rem 0 1.2rem 0;">
            <div style="font-size:2rem; line-height:1;">🎓</div>
            <div style="font-size:1.15rem; font-weight:700; color:#f1f5f9; margin-top:.4rem;">
                Student Success<br>Predictor
            </div>
            <div style="font-size:.72rem; color:#64748b; margin-top:.2rem; text-transform:uppercase; letter-spacing:.08em;">
                ML Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)
        page = st.radio(
            "",
            [
                "🏠 Présentation du projet",
                "🤖 Modélisation",
                "🎯 Démonstration",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Student Performance Dataset — UCI ML Repository")
        st.caption("Albert School — Projet ML 2025-2026")

    # ── Load data & models
    d          = _load_models()
    lr         = d["lr"]
    rf         = d["rf"]
    gb         = d["gb"]
    X_train_s  = d["X_train_s"]
    X_train_r  = d["X_train_r"]
    X_test_s   = d["X_test_s"]
    X_test_r   = d["X_test_r"]
    y_train         = d["y_train"]
    y_test          = d["y_test"]
    features_r      = d["features_r"]
    kmeans          = d["kmeans"]
    clust_scaler    = d["clust_scaler"]
    cluster_feats   = d["cluster_feats"]
    cluster_centers = d["cluster_centers"]
    cluster_labels  = d["cluster_labels"]

    # Pre-compute predictions for all models
    models_cfg = {
        "Régression Logistique": (lr, X_test_s),
        "Random Forest":         (rf, X_test_r),
        "Gradient Boosting":     (gb, X_test_r),
    }
    col_map = {
        "Régression Logistique": COLORS["lr"],
        "Random Forest":         COLORS["rf"],
        "Gradient Boosting":     COLORS["gb"],
    }
    results: dict[str, dict] = {}
    for name, (model, X_te) in models_cfg.items():
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        results[name] = {**_metrics(y_test, y_pred, y_proba),
                         "pred": y_pred, "proba": y_proba}

    # ══════════════════════════════════════════════════════════
    # PAGE 1 — PRÉSENTATION + EDA + BUSINESS
    # ══════════════════════════════════════════════════════════
    if page == "🏠 Présentation du projet":
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">Projet ML — Albert School 2025-2026</div>
            <p class="hero-title">🎓 Prédiction de la réussite scolaire</p>
            <p class="hero-sub">Identifier les élèves à risque d'échec dès la rentrée, avant les premières évaluations.</p>
        </div>
        """, unsafe_allow_html=True)

        df, df_mat, df_por = _load_raw()

        # ── Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Élèves",           "1 044")
        c2.metric("Variables brutes", "30")
        c3.metric("Taux de réussite", "78 %")
        c4.metric("Taux d'échec",     "22 %")
        st.divider()

        # ── 1. Présentation
        st.markdown('<p class="section-title">1 — Le projet</p>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2], gap="large")
        with col_l:
            st.markdown("""
Chaque année, des élèves décrochent **sans que personne ne l'ait anticipé**.
Les conseils de classe, les examens de mi-année — tout arrive trop tard.

Ce projet construit un système capable de **prédire la réussite d'un élève dès la rentrée**,
à partir de son profil personnel et comportemental, avant même la première note.

**Ce que ça permet :**
- Identifier les élèves à risque dès septembre
- Déclencher un accompagnement ciblé avant qu'ils ne décrochent
- Prioriser les ressources pédagogiques là où c'est vraiment nécessaire
            """)
        with col_r:
            st.markdown("""
<div class="card">
<strong>Source</strong> : Student Performance Dataset — UCI ML Repository<br>
<strong>Collecte</strong> : 2 lycées portugais, 2006-2007<br>
<strong>Cours</strong> : Mathématiques + Portugais<br><br>
<span class="tag">30 variables</span>
<span class="tag">Aucune valeur manquante</span>
<span class="tag">Licence académique</span>
</div>
            """, unsafe_allow_html=True)
            st.markdown("""
> Les notes intermédiaires **G1 et G2 sont exclues** : elles ne sont pas
> disponibles en début d'année (corrélation > 0.8 avec G3).
            """)
        st.divider()

        # ── 2. EDA — Pourquoi ce dataset ?
        st.markdown('<p class="section-title">2 — Exploration des données</p>', unsafe_allow_html=True)
        tab_cible, tab_num, tab_cat = st.tabs(
            ["📊 Variable cible", "📈 Corrélations numériques", "🗂️ Variables catégorielles"]
        )

        with tab_cible:
            col1, col2 = st.columns([3, 2], gap="large")
            with col1:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].hist(df_mat["G3"], bins=20, alpha=0.8,
                             color=COLORS["lr"], label="Mathématiques")
                axes[0].hist(df_por["G3"], bins=20, alpha=0.6,
                             color=COLORS["rf"], label="Portugais")
                axes[0].axvline(10, color="black", linestyle="--", lw=2, label="Seuil réussite (10)")
                axes[0].set_title("Distribution des notes finales G3", fontweight="bold")
                axes[0].set_xlabel("Note finale G3")
                axes[0].set_ylabel("Nombre d'élèves")
                axes[0].legend(fontsize=8)
                axes[0].grid(axis="y", alpha=0.3)
                counts = df["pass"].value_counts().sort_index()
                axes[1].pie(
                    counts,
                    labels=["Échec (G3 < 10)", "Réussite (G3 ≥ 10)"],
                    colors=[COLORS["Échec"], COLORS["Réussite"]],
                    autopct="%1.1f%%", startangle=90,
                    textprops={"fontsize": 11}, pctdistance=0.75,
                    wedgeprops={"linewidth": 2, "edgecolor": "white"},
                )
                axes[1].set_title("Répartition des classes", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            with col2:
                st.markdown("#### Ce qu'on observe")
                st.markdown("""
La distribution de G3 est **bimodale** :
- Un pic à **0** — élèves qui abandonnent ou échouent complètement
- Un pic autour de **12-13** — réussites classiques

Le seuil à 10 produit un déséquilibre naturel : **78 % de réussite vs 22 % d'échec**.
Ce déséquilibre est similaire en Maths et en Portugais — ce qui justifie de
**concaténer les deux datasets** et d'utiliser une métrique adaptée.
                """)
                rate = df.groupby("course")["pass"].mean() * 100
                ca, cb = st.columns(2)
                ca.metric("Réussite Maths",    f"{rate.get('math', 0):.1f} %")
                cb.metric("Réussite Portugais", f"{rate.get('portuguese', 0):.1f} %")

        with tab_num:
            col1, col2 = st.columns([3, 2], gap="large")
            with col1:
                num_cols = ["age", "traveltime", "studytime", "failures",
                            "famrel", "freetime", "goout", "health", "absences"]
                corr = df[num_cols + ["pass"]].corr()["pass"].drop("pass").sort_values()
                fig, ax = plt.subplots(figsize=(8, 5))
                colors_bar = [COLORS["Échec"] if v < 0 else COLORS["Réussite"]
                              for v in corr.values]
                bars = ax.barh(corr.index, corr.values, color=colors_bar,
                               edgecolor="white", height=0.65)
                ax.axvline(0, color="black", lw=1)
                for bar, val in zip(bars, corr.values):
                    ax.text(val + (0.003 if val >= 0 else -0.003),
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:+.2f}", va="center",
                            ha="left" if val >= 0 else "right", fontsize=8)
                ax.set_title("Corrélation de Pearson avec la réussite (G1/G2 exclus)",
                             fontweight="bold")
                ax.set_xlabel("Coefficient de corrélation")
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            with col2:
                st.markdown("#### Signaux identifiés")
                st.markdown("""
| Variable | Corrélation | Signal |
|---|---|---|
| `failures` | −0.37 | Le plus fort prédicteur négatif |
| `studytime` | +0.19 | Engagement scolaire direct |
| `goout` | −0.17 | Désengagement social |
| `absences` | −0.10 | Décrochage progressif |

Ces observations ont **directement motivé** le feature engineering :
`risk_score`, `study_vs_social`, `alc_total`…
                """)

        with tab_cat:
            col1, col2 = st.columns([3, 2], gap="large")
            with col1:
                cat_select = st.selectbox(
                    "Variable à explorer",
                    ["higher", "internet", "failures_cat", "studytime",
                     "sex", "address", "schoolsup"],
                    format_func=lambda x: {
                        "higher": "Ambition études sup.", "internet": "Accès internet",
                        "failures_cat": "Échecs passés", "studytime": "Temps d'étude",
                        "sex": "Sexe", "address": "Zone (urbain/rural)",
                        "schoolsup": "Soutien scolaire",
                    }.get(x, x)
                )
                plot_col = (
                    df["failures"].clip(upper=3).astype(str).replace("3", "3+")
                    if cat_select == "failures_cat"
                    else df[cat_select].astype(str)
                )
                pass_rate = df.groupby(plot_col)["pass"].mean() * 100
                fig, ax = plt.subplots(figsize=(7, 4))
                bar_colors = [COLORS["Réussite"] if v >= 70 else
                              COLORS["Échec"] if v < 60 else COLORS["lr"]
                              for v in pass_rate.values]
                bars = ax.bar(pass_rate.index, pass_rate.values,
                              color=bar_colors, edgecolor="white", linewidth=1.5, width=0.6)
                ax.axhline(78, color="gray", linestyle="--", lw=1.2, label="Moyenne (78 %)")
                ax.set_title(f"Taux de réussite par '{cat_select}'", fontweight="bold")
                ax.set_ylabel("% de réussite"); ax.set_ylim(0, 115)
                ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                            f"{bar.get_height():.0f}%",
                            ha="center", fontsize=9, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close()
            with col2:
                st.markdown("#### Pourquoi cette variable ?")
                INTERP_CAT = {
                    "higher": "Les élèves qui veulent faire des études supérieures réussissent bien plus souvent — c'est le facteur comportemental protecteur le plus fort.",
                    "internet": "L'accès à internet à domicile facilite les révisions et l'autonomie scolaire.",
                    "failures_cat": "Chaque échec passé réduit drastiquement les chances de réussir — c'est le prédicteur numéro 1.",
                    "studytime": "Plus le temps d'étude augmente, plus le taux de réussite grimpe — relation quasi-linéaire.",
                    "sex": "Les différences entre filles et garçons sont présentes mais modérées.",
                    "address": "Les élèves en zone urbaine ont légèrement plus de ressources (internet, bibliothèques).",
                    "schoolsup": "Le soutien scolaire est souvent demandé par les élèves en difficulté — corrélation inverse attendue.",
                }
                st.info(INTERP_CAT.get(cat_select, ""))

        st.divider()

        # ── 3. Application business
        st.markdown('<p class="section-title">3 — Application business</p>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3, gap="medium")
        with b1:
            st.markdown("""
<div class="card card-green">
<strong>Détection précoce</strong><br><br>
En début d'année scolaire, l'outil identifie les élèves
dont le profil correspond à un risque d'échec élevé,
<em>avant</em> les premières évaluations.
</div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown("""
<div class="card card-orange">
<strong>Priorisation des ressources</strong><br><br>
Les conseillers pédagogiques peuvent concentrer leur
accompagnement sur les profils les plus à risque,
optimisant l'impact de chaque intervention.
</div>
            """, unsafe_allow_html=True)
        with b3:
            st.markdown("""
<div class="card card-red">
<strong>Suivi comportemental</strong><br><br>
Les facteurs clés (absences, alcool, isolement social)
sont monitorés pour détecter les glissements progressifs
vers le décrochage.
</div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # PAGE 2 — MODÉLISATION
    # ══════════════════════════════════════════════════════════
    elif page == "🤖 Modélisation":
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">Machine Learning</div>
            <p class="hero-title">🤖 Modélisation</p>
            <p class="hero-sub">Comparaison des 3 modèles, métriques, interprétation et analyse des erreurs.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Résumé des 3 modèles
        st.markdown('<p class="section-title">1 — Les 3 modèles</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3, gap="medium")
        with m1:
            st.markdown("""
<div class="card">
<strong>Régression Logistique</strong> <span class="tag">Baseline</span><br><br>
Modèle linéaire interprétable. Les coefficients indiquent
directement l'impact de chaque variable sur la probabilité
de réussite. Régularisation L2, <code>class_weight='balanced'</code>.
</div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
<div class="card">
<strong>Random Forest</strong> <span class="tag">Ensemble</span><br><br>
200 arbres indépendants agrégés. Robuste aux non-linéarités
et à la multicolinéarité. Fournit une mesure d'importance
des features exploitable pour valider le feature engineering.
</div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown("""
<div class="card">
<strong>Gradient Boosting</strong> <span class="tag">SOTA</span><br><br>
Boosting séquentiel : chaque arbre corrige les erreurs du
précédent. État de l'art sur les données tabulaires.
<code>learning_rate=0.05</code>, <code>max_depth=3</code>.
</div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Métriques & comparaison
        st.markdown('<p class="section-title">2 — Comparaison des performances</p>', unsafe_allow_html=True)
        col_expl, col_table = st.columns([2, 3], gap="large")
        with col_expl:
            st.markdown("""
**Métrique principale : F1-score macro**

Avec 78 % de réussite et 22 % d'échec, l'accuracy est
trompeuse — un modèle qui prédit toujours "Réussite"
obtiendrait **78 % sans rien apprendre**.

Le F1 macro traite les deux classes à égalité :
> F1 macro = (F1_échec + F1_réussite) / 2

Il pénalise les élèves en difficulté non détectés.

**Métriques secondaires :**
- F1 Échec — capacité à détecter la classe minoritaire
- AUC-ROC — robustesse au seuil de décision
- Accuracy — lisibilité générale
            """)
        with col_table:
            df_metrics = pd.DataFrame(
                {name: {k: v for k, v in m.items() if k not in ("pred", "proba")}
                 for name, m in results.items()}
            ).T
            best_model = df_metrics["F1 macro ★"].idxmax()
            st.dataframe(df_metrics.round(3), use_container_width=True)
            st.success(f"Meilleur modèle : **{best_model}** — F1 macro = {df_metrics.loc[best_model, 'F1 macro ★']:.3f}")

            st.markdown("---")
            st.markdown("**Optimisation du seuil de décision**")
            st.caption("Par défaut le seuil est 0.5. Trouver le seuil optimal peut améliorer le F1 macro.")
            thresh_model = st.selectbox("Modèle", list(results.keys()), key="thresh_sel")
            y_proba_opt = results[thresh_model]["proba"]
            best_t, best_f1 = _best_threshold(y_test, y_proba_opt)
            f1_default = f1_score(y_test, (y_proba_opt >= 0.5).astype(int), average="macro")

            thresholds_range = np.linspace(0.1, 0.9, 81)
            f1_curve = [f1_score(y_test, (y_proba_opt >= t).astype(int), average="macro")
                        for t in thresholds_range]
            fig_t, ax_t = plt.subplots(figsize=(8, 3))
            ax_t.plot(thresholds_range, f1_curve, color=COLORS["lr"], lw=2)
            ax_t.axvline(0.5,   color="gray",         linestyle="--", lw=1.5, label=f"Seuil 0.5 → F1 = {f1_default:.3f}")
            ax_t.axvline(best_t, color=COLORS["Réussite"], linestyle="--", lw=1.5, label=f"Seuil optimal {best_t:.2f} → F1 = {best_f1:.3f}")
            ax_t.set_xlabel("Seuil de décision"); ax_t.set_ylabel("F1 macro")
            ax_t.set_title(f"F1 macro en fonction du seuil — {thresh_model}", fontweight="bold")
            ax_t.legend(fontsize=9); ax_t.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_t); plt.close()
            gain = (best_f1 - f1_default) * 100
            st.info(f"Gain potentiel en passant de 0.5 à {best_t:.2f} : **+{gain:.1f} points de F1 macro**")

        names_list  = list(results.keys())
        short_names = ["LR", "RF", "GB"]
        metrics_bar = ["F1 macro ★", "F1 Échec", "F1 Réussite", "Accuracy", "AUC-ROC"]
        palette     = ["#3b82f6", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"]

        col_bar, col_roc = st.columns(2, gap="medium")
        with col_bar:
            fig_bar = go.Figure()
            for metric, color in zip(metrics_bar, palette):
                fig_bar.add_trace(go.Bar(
                    name=metric,
                    x=short_names,
                    y=[results[n][metric] for n in names_list],
                    marker_color=color,
                    text=[f"{results[n][metric]:.3f}" for n in names_list],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                barmode="group", title="Métriques par modèle",
                yaxis=dict(range=[0, 1.15], title="Score"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, sans-serif"),
                height=420, margin=dict(t=50, b=10),
            )
            fig_bar.update_xaxes(showgrid=False)
            fig_bar.update_yaxes(gridcolor="#f1f5f9")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_roc:
            fig_roc = go.Figure()
            roc_colors = [COLORS["lr"], COLORS["rf"], COLORS["gb"]]
            for name, color in zip(names_list, roc_colors):
                fpr, tpr, _ = roc_curve(y_test, results[name]["proba"])
                auc = results[name]["AUC-ROC"]
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{name.split()[0]} — AUC={auc:.3f}",
                    line=dict(color=color, width=2.5),
                    hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
            ))
            fig_roc.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                              fillcolor="#f8fafc", opacity=0.3, line_width=0)
            fig_roc.update_layout(
                title="Courbes ROC — Test set",
                xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, sans-serif"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
                height=420, margin=dict(t=50, b=10),
            )
            fig_roc.update_xaxes(showgrid=True, gridcolor="#f1f5f9", range=[0, 1])
            fig_roc.update_yaxes(showgrid=True, gridcolor="#f1f5f9", range=[0, 1])
            st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()

        # ── Interprétation + Prédictions vs Réels (tabs)
        st.markdown('<p class="section-title">3 — Interprétation & analyse des erreurs</p>', unsafe_allow_html=True)
        tab_interp, tab_fe, tab_surprises, tab_shap, tab_errors = st.tabs(
            ["🔎 Interprétation du modèle", "🛠️ Variables créées & encodées", "💡 Surprises du feature engineering", "⚡ SHAP — explication individuelle", "📉 Prédictions vs Réel"]
        )

        with tab_interp:
            model_choice = st.selectbox(
                "Modèle à interpréter",
                ["Gradient Boosting", "Random Forest", "Régression Logistique"],
            )
            col_chart, col_read = st.columns([3, 2], gap="large")
            if model_choice == "Régression Logistique":
                coef = pd.Series(lr.coef_[0], index=X_train_s.columns).sort_values()
                top_n = st.slider("Nombre de features affichées", 10, len(coef), 20, key="lr_slider")
                coef_show = pd.concat([coef.head(top_n // 2), coef.tail(top_n // 2)])
                with col_chart:
                    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.28)))
                    colors_coef = [COLORS["Échec"] if v < 0 else COLORS["Réussite"]
                                   for v in coef_show.values]
                    ax.barh(coef_show.index, coef_show.values,
                            color=colors_coef, edgecolor="white", height=0.7)
                    ax.axvline(0, color="black", lw=1)
                    ax.set_title(f"Top {top_n} coefficients — Régression Logistique\n(rouge = facteur d'échec  |  vert = facteur protecteur)",
                                 fontweight="bold")
                    ax.set_xlabel("Coefficient (données scalées)")
                    ax.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()
                with col_read:
                    st.markdown("""
**Comment lire ce graphique ?**

Chaque barre = impact d'une variable sur la probabilité
de réussite, toutes les autres étant maintenues constantes.

- **Barre verte** → augmente la probabilité de réussite
- **Barre rouge** → augmente le risque d'échec

Les coefficients sont sur données **scalées** :
ils sont directement comparables entre eux.
                    """)
            else:
                model_obj  = rf if model_choice == "Random Forest" else gb
                color_imp  = COLORS["rf"] if model_choice == "Random Forest" else COLORS["gb"]
                INTERP = {
                    "risk_score":      "Score composite : failures×2 + absences + alcool − motivation",
                    "failures":        "Échecs passés — prédicteur n°1 (corrélation −0.37)",
                    "higher":          "Ambition études supérieures — facteur protecteur fort",
                    "absences":        "Absences → désengagement progressif",
                    "age":             "Reflète en partie le redoublement",
                    "studytime":       "Temps d'étude — signal comportemental positif",
                    "parent_edu":      "Capital éducatif familial moyen",
                    "study_vs_social": "Arbitrage temps d'étude vs vie sociale",
                    "alc_total":       "Consommation d'alcool totale",
                    "family_capital":  "Parents éduqués dans un contexte familial sain",
                }
                imp = pd.Series(model_obj.feature_importances_, index=features_r).sort_values(ascending=False)
                top_n = st.slider("Nombre de features affichées", 5, len(imp), 15, key="tree_slider")
                with col_chart:
                    imp_show = imp.head(top_n).iloc[::-1]
                    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.32)))
                    ax.barh(imp_show.index, imp_show.values,
                            color=color_imp, edgecolor="white", alpha=0.88, height=0.7)
                    ax.set_title(f"Top {top_n} features — {model_choice}",
                                 fontweight="bold")
                    ax.set_xlabel("Importance (réduction d'impureté)")
                    ax.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()
                with col_read:
                    st.markdown("**Top 5 — lecture**")
                    for feat, score in imp.head(5).items():
                        desc = INTERP.get(feat, "Variable comportementale ou familiale")
                        st.markdown(f"**`{feat}`** ({score:.3f})\n\n{desc}\n")

        with tab_fe:
            df_raw, _, _ = _load_raw()

            st.markdown("### Feature Engineering — variables créées")
            st.markdown("À partir des 30 variables brutes, 9 nouvelles features ont été construites :")

            fe_data = {
                "Feature créée":    ["alc_total", "alc_high_risk", "parent_edu",
                                     "study_vs_social", "motivated_with_resources",
                                     "family_capital", "has_support", "digital_access", "risk_score"],
                "Construction":     ["Dalc + Walc", "alc_total ≥ 5 (0/1)", "(Medu + Fedu) / 2",
                                     "studytime − goout", "higher='yes' ET internet='yes'",
                                     "parent_edu × famrel", "schoolsup='yes' OU famsup='yes'",
                                     "address='U' ET internet='yes'",
                                     "failures×2 + alc_high_risk + (absences>10) + (studytime=1) − higher"],
                "Variables supprimées": ["Dalc, Walc", "—", "Medu, Fedu",
                                         "—", "—", "—", "—", "—", "—"],
                "Justification":    [
                    "Dalc et Walc mesurent le même comportement (r > 0.6) — un score unique réduit la redondance",
                    "Seuil comportemental : risque élevé si alcool ≥ 50 % du maximum",
                    "Capital éducatif familial moyen — Medu/Fedu corrélés (r ≈ 0.6)",
                    "Capture l'arbitrage entre engagement scolaire et vie sociale",
                    "Motivation + ressources numériques : combinaison plus prédictive que les deux séparément",
                    "Des parents éduqués dans un foyer aux relations dégradées ont un impact limité",
                    "Présence d'au moins une source de soutien actif",
                    "Élève urbain avec internet = meilleures ressources de travail à domicile",
                    "Score composite des facteurs d'échec — failures pondéré ×2 car corrélation −0.37",
                ],
            }
            st.dataframe(pd.DataFrame(fe_data), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Encodage — comment les variables ont été transformées")

            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown("**Binaires yes/no → 0/1**")
                st.markdown("""
| Variable | 0 | 1 |
|---|---|---|
| schoolsup | non | oui |
| famsup | non | oui |
| paid | non | oui |
| activities | non | oui |
| nursery | non | oui |
| higher | non | oui |
| internet | non | oui |
| romantic | non | oui |
                """)
            with c2:
                st.markdown("**Binaires 2 valeurs → 0/1**")
                st.markdown("""
| Variable | 0 | 1 |
|---|---|---|
| sex | M | F |
| address | R (rural) | U (urbain) |
| famsize | LE3 | GT3 |
| Pstatus | A (séparé) | T (ensemble) |
| school | MS | GP |
                """)
            with c3:
                st.markdown("**Nominales → One-Hot Encoding**")
                st.markdown("""
| Variable | Colonnes créées |
|---|---|
| Mjob | Mjob_teacher, Mjob_health… |
| Fjob | Fjob_teacher, Fjob_health… |
| reason | reason_home, reason_rep… |
| guardian | guardian_mother… |
| course | course_math, course_por |

*drop_first=False pour garder toutes les modalités*
                """)

            st.divider()
            st.markdown("### Avant / Après — aperçu des données")
            col_before, col_after = st.columns(2, gap="medium")
            with col_before:
                st.markdown("**Données brutes (5 premières lignes)**")
                raw_cols = ["sex", "age", "Mjob", "failures", "Dalc", "Walc",
                            "Medu", "Fedu", "higher", "absences", "G3"]
                st.dataframe(df_raw[raw_cols].head(), use_container_width=True, hide_index=True)
            with col_after:
                st.markdown("**Après engineering + encodage + scaling**")
                show_cols = ["sex", "age", "failures", "alc_total", "parent_edu",
                             "risk_score", "study_vs_social", "higher", "absences"]
                show_cols_present = [c for c in show_cols if c in X_test_s.columns]
                st.dataframe(X_test_s[show_cols_present].head().round(3),
                             use_container_width=True, hide_index=True)

            st.caption(f"Dataset final : {X_train_s.shape[1]} features · {X_train_s.shape[0] + X_test_s.shape[0]} élèves · split 80/20 · StandardScaler fit sur train uniquement")

        with tab_surprises:
            st.markdown("""
Lors du feature engineering, on a créé des variables en pensant qu'elles auraient
un fort impact prédictif. Voici ce que les modèles arborescents ont vraiment révélé.
            """)
            model_surp = st.selectbox("Modèle", ["Gradient Boosting", "Random Forest"],
                                      key="surp_model")
            model_obj_s = gb if model_surp == "Gradient Boosting" else rf
            imp_s = pd.Series(model_obj_s.feature_importances_, index=features_r).sort_values(ascending=False)
            imp_pct = imp_s / imp_s.sum() * 100

            # Features engineered qu'on espérait importantes
            ENGINEERED = {
                "has_support":              "Support scolaire ou familial (schoolsup OR famsup)",
                "digital_access":           "Élève urbain avec internet (address AND internet)",
                "alc_high_risk":            "Alcool à risque élevé (alc_total ≥ 5, binaire)",
                "motivated_with_resources": "Ambition + internet (higher AND internet)",
                "family_capital":           "Capital familial (parent_edu × famrel)",
                "study_vs_social":          "Étude vs vie sociale (studytime − goout)",
                "alc_total":               "Consommation totale d'alcool (Dalc + Walc)",
                "parent_edu":              "Éducation parentale moyenne (Medu + Fedu) / 2",
                "risk_score":              "Score de risque composite",
            }

            eng_imp = {k: imp_pct.get(k, 0) for k in ENGINEERED if k in imp_pct.index}
            eng_series = pd.Series(eng_imp).sort_values()

            col_g, col_txt = st.columns([3, 2], gap="large")
            with col_g:
                fig, ax = plt.subplots(figsize=(8, 5))
                threshold = imp_pct.mean()
                bar_colors = [
                    COLORS["Réussite"] if v >= threshold else
                    "#F39C12" if v >= threshold * 0.5 else
                    COLORS["Échec"]
                    for v in eng_series.values
                ]
                bars = ax.barh(eng_series.index, eng_series.values,
                               color=bar_colors, edgecolor="white", height=0.65)
                ax.axvline(threshold, color="gray", linestyle="--", lw=1.5,
                           label=f"Importance moyenne ({threshold:.2f} %)")
                for bar, val in zip(bars, eng_series.values):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:.2f} %", va="center", fontsize=8)
                ax.set_title(f"Importance des features engineered — {model_surp}",
                             fontweight="bold")
                ax.set_xlabel("% de l'importance totale du modèle")
                ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col_txt:
                st.markdown("**Ce qu'on a appris**")
                bottom_3 = eng_series.head(3)
                top_3    = eng_series.tail(3)
                st.error("**Décevantes** — impact quasi nul sur le modèle")
                for feat, val in bottom_3.items():
                    st.markdown(f"- `{feat}` → **{val:.2f} %** — {ENGINEERED[feat]}")
                st.success("**Utiles** — au-dessus de la moyenne")
                for feat, val in top_3.items():
                    st.markdown(f"- `{feat}` → **{val:.2f} %** — {ENGINEERED[feat]}")
                st.markdown("""
---
**Pourquoi ces features déçoivent-elles ?**

`has_support` et `digital_access` capturent des conditions
déjà encodées dans `schoolsup`, `famsup`, `address` et
`internet` — le modèle y accède directement et n'a pas besoin
de leur combinaison.

`alc_high_risk` est redondant avec `alc_total` : l'arbre
peut lui-même trouver le seuil optimal, il n'a pas besoin
qu'on le précode en binaire.
                """)

        with tab_shap:
            st.markdown("""
**SHAP (SHapley Additive exPlanations)** décompose la prédiction d'un élève individuel :
pour chaque variable, il mesure combien elle pousse la probabilité vers la réussite ou vers l'échec.
            """)
            shap_model_choice = st.selectbox(
                "Modèle", ["Gradient Boosting", "Random Forest"], key="shap_model"
            )
            shap_idx = st.slider(
                "Élève à analyser (index dans le test set)", 0, len(X_test_r) - 1, 0,
                key="shap_idx"
            )
            shap_model_obj = gb if shap_model_choice == "Gradient Boosting" else rf

            with st.spinner("Calcul des valeurs SHAP..."):
                explainer = (_shap_explainer_gb if shap_model_choice == "Gradient Boosting"
                             else _shap_explainer_rf)(shap_model_obj, X_train_r)
                shap_vals = explainer(X_test_r.iloc[[shap_idx]])
                sv = shap_vals[0].values
                if sv.ndim == 2:
                    sv = sv[:, 1]
                base = shap_vals[0].base_values
                if hasattr(base, '__len__'):
                    base = base[1]
                feat_names = list(X_test_r.columns)

            y_true_val  = int(y_test.iloc[shap_idx])
            y_pred_val  = int(shap_model_obj.predict(X_test_r.iloc[[shap_idx]])[0])
            y_proba_val = float(shap_model_obj.predict_proba(X_test_r.iloc[[shap_idx]])[0, 1])

            col_info, col_chart = st.columns([1, 3], gap="large")
            with col_info:
                st.markdown("**Profil de l'élève**")
                real_icon = "🟢 Réussite" if y_true_val == 1 else "🔴 Échec"
                pred_icon = "🟢 Réussite" if y_pred_val == 1 else "🔴 Échec"
                st.metric("Réalité",    real_icon)
                st.metric("Prédiction", pred_icon)
                st.metric("P(Réussite)", f"{y_proba_val * 100:.1f} %")
                correct = y_true_val == y_pred_val
                if correct:
                    st.success("Bonne prédiction")
                else:
                    st.error("Mauvaise prédiction")

            with col_chart:
                shap_series = pd.Series(sv, index=feat_names)
                top_pos = shap_series.nlargest(8)
                top_neg = shap_series.nsmallest(8)
                show    = pd.concat([top_neg, top_pos]).sort_values()

                fig_shap = go.Figure(go.Bar(
                    x=show.values,
                    y=show.index,
                    orientation="h",
                    marker_color=[COLORS["Réussite"] if v > 0 else COLORS["Échec"]
                                  for v in show.values],
                    text=[f"{v:+.3f}" for v in show.values],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>SHAP = %{x:+.4f}<extra></extra>",
                ))
                fig_shap.add_vline(x=0, line_width=1.5, line_color="#374151")
                fig_shap.update_layout(
                    title=f"SHAP — Élève #{shap_idx} | base={float(base):.2f} → P(réussite)={y_proba_val:.2f}",
                    xaxis_title="Impact sur la prédiction (SHAP)",
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter, sans-serif", size=12),
                    height=450, margin=dict(t=50, b=20, l=160, r=80),
                )
                fig_shap.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
                fig_shap.update_yaxes(showgrid=False)
                st.plotly_chart(fig_shap, use_container_width=True)

            st.caption("Les valeurs SHAP s'additionnent à partir de la valeur de base (prédiction moyenne) pour donner la prédiction finale.")

        with tab_errors:
            model_choice_2 = st.selectbox(
                "Modèle", ["Gradient Boosting", "Random Forest", "Régression Logistique"],
                key="errors_model",
            )
            y_pred_sel  = results[model_choice_2]["pred"]
            y_proba_sel = results[model_choice_2]["proba"]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sel).ravel()

            col1, col2, col3 = st.columns([2, 2, 2], gap="medium")
            with col1:
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ConfusionMatrixDisplay.from_predictions(
                    y_test, y_pred_sel,
                    display_labels=["Échec", "Réussite"],
                    cmap="Blues", ax=ax, colorbar=False,
                )
                ax.set_title(f"Matrice de confusion\n{model_choice_2}", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col2:
                st.markdown("#### Décomposition des erreurs")
                st.markdown(f"""
| Verdict | Nombre |
|---|---|
| Réussites détectées (VP) | **{tp}** |
| Échecs détectés (VN) | **{tn}** |
| Échecs prédits Réussite (FN) | **{fn}** |
| Réussites prédites Échec (FP) | **{fp}** |

**Taux de détection de l'échec** : {tn / (tn + fp) * 100:.0f} %

Les **Faux Négatifs** ({fn}) sont les plus critiques :
ce sont des élèves en difficulté que le modèle
n'a pas identifiés.
                """)

            with col3:
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ax.hist(y_proba_sel[y_test == 0], bins=20, alpha=0.75,
                        color=COLORS["Échec"], label="Réels Échec", density=True)
                ax.hist(y_proba_sel[y_test == 1], bins=20, alpha=0.65,
                        color=COLORS["Réussite"], label="Réels Réussite", density=True)
                ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Seuil 0.5")
                ax.set_title("Distribution des probabilités\nprédites", fontweight="bold")
                ax.set_xlabel("P(Réussite)")
                ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

    # ══════════════════════════════════════════════════════════
    # PAGE 3 — DÉMONSTRATION
    # ══════════════════════════════════════════════════════════
    elif page == "🎯 Démonstration":
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">Simulateur temps réel</div>
            <p class="hero-title">🎯 Simuler un nouvel élève</p>
            <p class="hero-sub">Renseignez le profil d'un élève entrant pour prédire sa probabilité de réussite en fin d'année.</p>
        </div>
        """, unsafe_allow_html=True)

        col_form, col_result = st.columns([3, 2], gap="large")

        with col_form:
            st.markdown("#### Profil de l'élève")
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                age      = st.slider("Âge", 15, 22, 16)
                failures = st.selectbox("Échecs passés", [0, 1, 2, 3],
                                        format_func=lambda x: f"{x} échec{'s' if x > 1 else ''}")
            with r1c2:
                studytime = st.selectbox(
                    "Temps d'étude / semaine", [1, 2, 3, 4],
                    format_func=lambda x: {1: "< 2h", 2: "2–5h", 3: "5–10h", 4: "> 10h"}[x],
                    index=1,
                )
                absences = st.slider("Absences (jours)", 0, 30, 3)
            with r1c3:
                goout  = st.slider("Sorties avec amis (1–5)", 1, 5, 2)
                famrel = st.slider("Relations familiales (1–5)", 1, 5, 4)

            st.markdown("---")
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                higher   = st.radio("Ambition études sup. ?", ["Oui", "Non"])
                internet = st.radio("Accès internet ?",        ["Oui", "Non"])
            with r2c2:
                Medu = st.slider("Éducation mère (0–4)", 0, 4, 2)
                Fedu = st.slider("Éducation père (0–4)", 0, 4, 2)
            with r2c3:
                Dalc = st.slider("Alcool semaine (1–5)", 1, 5, 1)
                Walc = st.slider("Alcool weekend (1–5)", 1, 5, 1)

            st.markdown("---")
            r3c1, r3c2 = st.columns(2)
            with r3c1:
                Mjob = st.selectbox("Profession mère",
                                    ["teacher", "health", "services", "at_home", "other"])
            with r3c2:
                Fjob = st.selectbox("Profession père",
                                    ["teacher", "health", "services", "at_home", "other"])

            predict_btn = st.button("Prédire la réussite", type="primary",
                                    use_container_width=True)

        with col_result:
            if predict_btn:
                higher_bin    = 1 if higher   == "Oui" else 0
                internet_bin  = 1 if internet == "Oui" else 0
                alc_total     = Dalc + Walc
                alc_high_risk = int(alc_total >= 5)
                parent_edu    = (Medu + Fedu) / 2
                study_vs_soc  = studytime - goout
                motivated     = int(higher_bin == 1 and internet_bin == 1)
                family_cap    = parent_edu * famrel
                digital_acc   = int(internet_bin == 1)
                risk_score    = max(0, failures * 2 + alc_high_risk
                                    + int(absences > 10) + int(studytime == 1) - higher_bin)

                feat_vec = {f: 0 for f in features_r}
                feat_vec.update({
                    "age": age, "traveltime": 1, "studytime": studytime,
                    "failures": failures, "famrel": famrel, "freetime": 3,
                    "goout": goout, "health": 3, "absences": absences,
                    "school": 1, "sex": 0, "address": internet_bin,
                    "famsize": 1, "Pstatus": 1,
                    "schoolsup": 0, "famsup": 0, "paid": 0,
                    "activities": 0, "nursery": 1,
                    "higher": higher_bin, "internet": internet_bin, "romantic": 0,
                    "alc_total": alc_total, "alc_high_risk": alc_high_risk,
                    "parent_edu": parent_edu, "study_vs_social": study_vs_soc,
                    "motivated_with_resources": motivated,
                    "family_capital": family_cap,
                    "has_support": 0, "digital_access": digital_acc,
                    "risk_score": risk_score,
                    f"Mjob_{Mjob}": 1, f"Fjob_{Fjob}": 1,
                    "reason_reputation": 1, "guardian_mother": 1, "course_math": 1,
                })
                X_new = pd.DataFrame([feat_vec])[features_r]

                proba_lr   = lr.predict_proba(X_new)[0, 1]
                proba_rf   = rf.predict_proba(X_new)[0, 1]
                proba_gb   = gb.predict_proba(X_new)[0, 1]
                proba_mean = float(np.mean([proba_lr, proba_rf, proba_gb]))

                # Verdict
                if proba_mean >= 0.65:
                    verdict_txt   = "Réussite probable"
                    verdict_color = COLORS["Réussite"]
                    verdict_icon  = "✅"
                elif proba_mean >= 0.4:
                    verdict_txt   = "Profil incertain"
                    verdict_color = "#F39C12"
                    verdict_icon  = "⚠️"
                else:
                    verdict_txt   = "Risque d'échec"
                    verdict_color = COLORS["Échec"]
                    verdict_icon  = "🚨"

                st.markdown(f"""
<div class="verdict-box" style="background:linear-gradient(135deg, {verdict_color}18, {verdict_color}08);
     border: 2px solid {verdict_color}55;">
    <div class="verdict-icon">{verdict_icon}</div>
    <div class="verdict-label" style="color:{verdict_color};">{verdict_txt}</div>
    <div class="verdict-prob" style="color:{verdict_color};">
        P(Réussite) = <strong>{proba_mean*100:.1f} %</strong>
    </div>
</div>
                """, unsafe_allow_html=True)

                # Jauge circulaire
                gauge_color = verdict_color
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba_mean * 100,
                    number={"suffix": " %", "font": {"size": 36, "color": gauge_color, "family": "Inter"}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%",
                                 "tickfont": {"size": 11}, "nticks": 6},
                        "bar": {"color": gauge_color, "thickness": 0.25},
                        "bgcolor": "white",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40],  "color": "#fef2f2"},
                            {"range": [40, 65], "color": "#fffbeb"},
                            {"range": [65, 100],"color": "#f0fdf4"},
                        ],
                        "threshold": {
                            "line": {"color": "#374151", "width": 3},
                            "thickness": 0.8, "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=230, margin=dict(t=20, b=10, l=30, r=30),
                    paper_bgcolor="white", font=dict(family="Inter, sans-serif"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Détail par modèle
                st.markdown("**Consensus des 3 modèles**")
                for name, proba in [("Régression Logistique", proba_lr),
                                     ("Random Forest",         proba_rf),
                                     ("Gradient Boosting",     proba_gb)]:
                    icon = "🟢" if proba >= 0.5 else "🔴"
                    st.progress(proba, text=f"{icon} {name}: {proba*100:.1f} %")

                st.divider()

                # ── Clustering — trouver le profil type
                student_clust_vec = np.array([[
                    failures, studytime, goout, absences,
                    alc_total, higher_bin, famrel,
                    (Medu + Fedu) / 2, risk_score,
                ]])
                student_clust_scaled = clust_scaler.transform(student_clust_vec)
                cluster_id = int(kmeans.predict(student_clust_scaled)[0])
                clabel, cdesc = cluster_labels[cluster_id]
                centroid = cluster_centers.iloc[cluster_id]

                st.markdown(f"""
<div class="card" style="border-left-color:{verdict_color}; margin-bottom:1rem;">
    <div style="font-size:1.1rem; font-weight:700;">{clabel}</div>
    <div style="color:#64748b; font-size:.9rem; margin-top:.3rem;">{cdesc}</div>
</div>
                """, unsafe_allow_html=True)

                # Radar chart profil élève
                st.markdown("**Radar — profil vs cluster**")
                radar_cats = ["Étude", "Social", "Famille", "Ambition", "Sobriété", "Assiduité"]
                radar_vals = [
                    studytime / 4,
                    1 - (goout - 1) / 4,
                    famrel / 5,
                    higher_bin,
                    1 - (alc_total - 2) / 8,
                    1 - min(absences, 30) / 30,
                ]
                radar_vals_pct = [max(0, min(1, v)) * 100 for v in radar_vals]
                def hex_to_rgba(hex_color, alpha=0.2):
                    h = hex_color.lstrip("#")
                    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    return f"rgba({r},{g},{b},{alpha})"

                # Centroïd du cluster (normalisé sur les mêmes axes)
                centroid_radar = [
                    centroid["studytime"] / 4,
                    1 - (centroid["goout"] - 1) / 4,
                    centroid["famrel"] / 5,
                    centroid["higher"],
                    1 - (centroid["alc_total"] - 2) / 8,
                    1 - min(centroid["absences"], 30) / 30,
                ]
                centroid_pct = [max(0, min(1, v)) * 100 for v in centroid_radar]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=centroid_pct + [centroid_pct[0]],
                    theta=radar_cats + [radar_cats[0]],
                    fill="toself",
                    fillcolor="rgba(148,163,184,0.15)",
                    line=dict(color="#94a3b8", width=1.5, dash="dot"),
                    name="Centroïde cluster",
                    hovertemplate="%{theta}: %{r:.0f}%<extra>Cluster</extra>",
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_vals_pct + [radar_vals_pct[0]],
                    theta=radar_cats + [radar_cats[0]],
                    fill="toself",
                    fillcolor=hex_to_rgba(verdict_color, 0.2),
                    line=dict(color=verdict_color, width=2.5),
                    name="Cet élève",
                    hovertemplate="%{theta}: %{r:.0f}%<extra>Élève</extra>",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100],
                                        ticksuffix="%", tickfont=dict(size=10)),
                        angularaxis=dict(tickfont=dict(size=12, family="Inter")),
                        bgcolor="white",
                    ),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter, sans-serif"),
                    height=300, margin=dict(t=20, b=20, l=40, r=40),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Facteurs de risque
                st.markdown("**Analyse du profil**")
                factors = [
                    ("Échecs passés",       failures,       failures > 0,      f"{failures}"),
                    ("Score de risque",     risk_score,     risk_score >= 3,    f"{risk_score}/7"),
                    ("Absences",            absences,       absences > 10,      f"{absences} j"),
                    ("Ambition études sup.",higher_bin,     not higher_bin,     "Oui" if higher_bin else "Non"),
                    ("Alcool (total)",      alc_total,      alc_high_risk,      f"{alc_total}/10"),
                    ("Étude vs Social",     study_vs_soc,   study_vs_soc < 0,   f"{study_vs_soc:+d}"),
                ]
                for label, _, is_risk, display in factors:
                    icon = "🔴" if is_risk else "🟢"
                    st.markdown(f"{icon} **{label}** : {display}")

                st.divider()

                # ── Score de confiance ─────────────────────────────────────
                probas_all = [proba_lr, proba_rf, proba_gb]
                std_models = float(np.std(probas_all))
                st.markdown("**Fiabilité de la prédiction**")
                if std_models < 0.05:
                    st.success(f"Consensus fort entre les 3 modèles (écart-type = {std_models:.3f}) — prédiction fiable")
                elif std_models < 0.12:
                    st.warning(f"Légère divergence entre modèles (écart-type = {std_models:.3f}) — profil ambigu")
                else:
                    st.error(f"Désaccord entre modèles (écart-type = {std_models:.3f}) — profil très incertain")

                st.divider()

                # ── Simulation "et si..." ──────────────────────────────────
                st.markdown("**Simulation — et si cet élève changeait ?**")

                feat_vec_if = feat_vec.copy()
                new_studytime = min(studytime + 1, 4)
                feat_vec_if["studytime"] = new_studytime
                feat_vec_if["study_vs_social"] = new_studytime - goout
                feat_vec_if["risk_score"] = max(0, feat_vec_if["risk_score"] - int(studytime == 1))
                X_if = pd.DataFrame([feat_vec_if])[features_r]
                p_if_gb = gb.predict_proba(X_if)[0, 1]
                delta_study = (p_if_gb - proba_gb) * 100
                arrow = "⬆️" if delta_study > 0 else "➡️"
                st.markdown(f"📚 +1 niveau d'étude/semaine → P(réussite) GB : {proba_gb*100:.1f}% → **{p_if_gb*100:.1f}%** ({arrow} {delta_study:+.1f} pts)")

                feat_vec_if2 = feat_vec.copy()
                new_abs = max(0, absences - 5)
                feat_vec_if2["absences"] = new_abs
                feat_vec_if2["risk_score"] = max(0, feat_vec_if2["risk_score"] - int(absences > 10) + int(new_abs > 10))
                X_if2 = pd.DataFrame([feat_vec_if2])[features_r]
                p_if_gb2 = gb.predict_proba(X_if2)[0, 1]
                delta_abs = (p_if_gb2 - proba_gb) * 100
                arrow2 = "⬆️" if delta_abs > 0 else "➡️"
                st.markdown(f"📅 −5 jours d'absence → P(réussite) GB : {proba_gb*100:.1f}% → **{p_if_gb2*100:.1f}%** ({arrow2} {delta_abs:+.1f} pts)")

                if failures > 0:
                    feat_vec_if3 = feat_vec.copy()
                    new_fail = max(0, failures - 1)
                    feat_vec_if3["failures"] = new_fail
                    feat_vec_if3["risk_score"] = max(0, feat_vec_if3["risk_score"] - 2)
                    X_if3 = pd.DataFrame([feat_vec_if3])[features_r]
                    p_if_gb3 = gb.predict_proba(X_if3)[0, 1]
                    delta_fail = (p_if_gb3 - proba_gb) * 100
                    arrow3 = "⬆️" if delta_fail > 0 else "➡️"
                    st.markdown(f"🎯 −1 échec passé → P(réussite) GB : {proba_gb*100:.1f}% → **{p_if_gb3*100:.1f}%** ({arrow3} {delta_fail:+.1f} pts)")

            else:
                st.markdown("""
<div class="card" style="text-align:center; padding: 2rem;">
<div style="font-size:3rem;">👤</div>
<div style="font-size:1.1rem; margin-top:.5rem; color:#555;">
Renseignez le profil de l'élève<br>et cliquez sur <strong>Prédire la réussite</strong>
</div>
</div>
                """, unsafe_allow_html=True)
                st.divider()
                st.markdown("**Exemples de profils**")
                st.success("**Profil favorable** : 0 échec · études sup. = Oui · studytime ≥ 3 · absences < 5 · alcool faible")
                st.error("**Profil à risque** : 2+ échecs · études sup. = Non · studytime = 1 · absences > 15 · alcool élevé")

if __name__ == "__main__":
    build_app()