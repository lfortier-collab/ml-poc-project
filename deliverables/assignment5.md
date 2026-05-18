# Assignment 5 — Interface Streamlit : Student Success Predictor

---

## 1. Description de l'application

**Student Success Predictor** est une application Streamlit interactive qui permet d'explorer, comprendre et utiliser trois modèles de machine learning entraînés pour prédire la réussite scolaire d'élèves portugais en fin d'année. L'application s'appuie sur le dataset UCI Student Performance (matières mathématiques et portugais, 1 044 élèves au total) et expose les résultats de manière visuelle et pédagogique.

---

## 2. Objectif de l'interface

L'interface poursuit trois objectifs complémentaires :

- **Exploration** : permettre à un utilisateur non-technique de comprendre les données, la distribution de la variable cible et les relations entre variables socio-comportementales et réussite.
- **Démonstration** : présenter les trois modèles entraînés (Régression Logistique, Random Forest, Gradient Boosting), comparer leurs performances et expliquer leurs prédictions via des outils d'interprétabilité (SHAP, coefficients, importances).
- **Prédiction interactive** : simuler en temps réel la probabilité de réussite d'un nouvel élève à partir d'un profil saisi manuellement.

---

## 3. Fonctionnalités implémentées

### Page 1 — Présentation du projet

- **Bannière d'accueil** avec métriques clés (nombre d'élèves, taux de réussite, nombre de features, nombre de modèles).
- **Section "Le projet"** : contexte métier, objectif de prédiction précoce (avant les notes intermédiaires), description du dataset.
- **Exploration des données (3 onglets)** :
  - *Distribution de la cible* : camembert et barres empilées montrant la répartition Réussite/Échec par matière.
  - *Variables numériques* : histogrammes comparatifs colorés par classe pour les features clés (absences, studytime, failures, goout, age...).
  - *Variables catégorielles* : taux de réussite par catégorie pour les variables binaires et nominales (higher, internet, Mjob, Fjob...).
- **Application business** : explication de l'usage pédagogique (détection précoce, déclenchement d'interventions ciblées).

### Page 2 — Modélisation

- **Présentation des 3 modèles** : description de la Régression Logistique, du Random Forest et du Gradient Boosting, avec leurs hyperparamètres clés.
- **Comparaison des performances** : tableau de métriques (F1 macro, F1 par classe, précision, rappel, accuracy) lu depuis `results/model_metrics.csv`, avec mise en évidence du meilleur score par colonne.
- **Visualisation des métriques** : graphique en barres groupées comparant les F1-scores des 3 modèles.
- **Courbes ROC** : courbes AUC pour chaque modèle sur le jeu de test.
- **Matrices de confusion** : une matrice par modèle, affichées côte à côte.
- **Interprétation & analyse des erreurs (5 onglets)** :
  - *Interprétation LR* : coefficients de la régression logistique sous forme de barres horizontales colorées par direction d'effet.
  - *Feature engineering* : visualisation de l'importance des features construites (`risk_score`, `family_capital`, `study_vs_social`...) dans le Random Forest.
  - *Cas surprenants* : analyse des faux positifs et faux négatifs, distribution des probabilités sur les erreurs, profil moyen des cas mal classifiés.
  - *Valeurs SHAP* : summary plot et waterfall plot SHAP pour le Gradient Boosting et le Random Forest — identification des variables les plus influentes par prédiction.
  - *Analyse des erreurs* : scatter plot probabilité prédite vs label réel, avec coloration des faux positifs/négatifs.

### Page 3 — Démonstration

- **Formulaire de saisie** du profil d'un élève (voir section Inputs).
- **Prédiction en temps réel** par les 3 modèles dès le clic sur le bouton.
- **Verdict agrégé** (moyenne des 3 probabilités) avec code couleur : Réussite probable / Profil incertain / Risque d'échec.
- **Jauge circulaire** (Plotly Indicator) affichant P(Réussite) en pourcentage.
- **Consensus des 3 modèles** : barres de progression individuelles pour chaque modèle.
- **Profil de clustering** : assignation de l'élève à un profil type via K-Means (Profil favorable / Profil à risque).
- **Radar chart** comparant le profil de l'élève saisi au centroïde de son cluster sur 6 dimensions (Étude, Social, Famille, Ambition, Sobriété, Assiduité).
- **Simulation "et si..."** : sliders permettant de modifier les variables comportementales clés pour observer l'évolution de la probabilité en temps réel.

---

## 4. Inputs utilisateurs

Les inputs sont tous sur la page **Démonstration**. Ils sont organisés en 3 rangées :

| Input | Type de widget | Plage / Options |
|---|---|---|
| Age | Slider | 15 – 22 |
| Échecs passés | Selectbox | 0, 1, 2, 3 |
| Temps d'étude / semaine | Selectbox | < 2h, 2–5h, 5–10h, > 10h |
| Absences (jours) | Slider | 0 – 30 |
| Sorties avec amis | Slider | 1 – 5 |
| Relations familiales | Slider | 1 – 5 |
| Ambition études supérieures | Radio | Oui / Non |
| Accès internet | Radio | Oui / Non |
| Éducation mère | Slider | 0 – 4 |
| Éducation père | Slider | 0 – 4 |
| Alcool semaine | Slider | 1 – 5 |
| Alcool weekend | Slider | 1 – 5 |
| Profession mère | Selectbox | teacher, health, services, at_home, other |
| Profession père | Selectbox | teacher, health, services, at_home, other |

Les features construites (`alc_total`, `risk_score`, `family_capital`, `study_vs_social`, etc.) sont calculées automatiquement à partir des inputs bruts avant la prédiction — l'utilisateur n'a pas à les renseigner.

---

## 5. Outputs affichés

| Output | Page | Description |
|---|---|---|
| Métriques clés (4 KPIs) | Présentation | Nombre d'élèves, taux de réussite, features, modèles |
| Graphiques EDA | Présentation | Camembert, histogrammes, barres de taux de réussite |
| Tableau de métriques | Modélisation | F1 macro/classe, précision, rappel, accuracy |
| Graphiques de performance | Modélisation | Barres groupées, courbes ROC, matrices de confusion |
| Coefficients LR | Modélisation | Barres horizontales colorées par effet positif/négatif |
| Importances RF | Modélisation | Top features par gain d'impureté |
| Valeurs SHAP | Modélisation | Summary plot et waterfall plot par modèle |
| Analyse des erreurs | Modélisation | Scatter, distributions des FP/FN |
| Verdict de prédiction | Démonstration | Bannière colorée + probabilité moyenne |
| Jauge Plotly | Démonstration | Indicateur circulaire 0–100 % |
| Consensus 3 modèles | Démonstration | Barres de progression par modèle |
| Profil clustering | Démonstration | Libellé et description du cluster K-Means |
| Radar chart | Démonstration | Profil élève vs centroïde du cluster |

---

## 6. Structure de l'application

```
src/app.py                  # Point d'entrée Streamlit (fonction build_app())
├── _load_models()          # Chargement des 3 modèles .joblib + données train/test
│                           # + K-Means clustering (2 clusters)
├── _load_raw()             # Chargement des splits scalés et bruts depuis data/processed/
├── _metrics()              # Calcul des métriques sur le jeu de test
├── _shap_explainer_gb()    # Explainer SHAP TreeExplainer (GB)
├── _shap_explainer_rf()    # Explainer SHAP TreeExplainer (RF)
├── _best_threshold()       # Recherche du seuil optimal en F1-macro
└── build_app()             # Fonction principale
    ├── Sidebar             # Navigation radio : 3 pages
    ├── Page "Présentation" # EDA + contexte métier
    ├── Page "Modélisation" # Comparaison modèles + interprétabilité
    └── Page "Démonstration"# Formulaire + prédiction temps réel
```

**Fichiers associés :**

| Fichier | Rôle |
|---|---|
| `src/app.py` | Application Streamlit (1 444 lignes) |
| `src/config.py` | Chemins, registre des modèles, configuration Streamlit |
| `src/data.py` | Pipeline de chargement et préparation des données |
| `src/metrics.py` | Calcul des métriques d'évaluation |
| `src/results.py` | Sauvegarde des métriques dans `results/model_metrics.csv` |
| `scripts/main.py` | Orchestration : évaluation + lancement Streamlit |
| `models/*.joblib` | Modèles entraînés sérialisés |
| `data/processed/*.csv` | Splits train/test (scalés et bruts) |
| `results/model_metrics.csv` | Métriques calculées à l'exécution |

---

## 7. Comment lancer l'application

### Prérequis

```bash
# Depuis la racine du projet
cd ml-poc-project
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

### Lancement complet (recommandé)

La commande suivante évalue les modèles, sauvegarde les métriques, puis lance Streamlit :

```bash
python scripts/main.py
```

L'application est alors accessible à l'adresse :

```
http://localhost:8501
```

### Lancement direct de Streamlit (sans réévaluation)

Si les modèles ont déjà été évalués et que `results/model_metrics.csv` existe :

```bash
PYTHONPATH=./src streamlit run src/app.py --server.address localhost --server.port 8501
```
