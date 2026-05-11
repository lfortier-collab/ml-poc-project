# Assignment 2 — Feature Engineering & Preprocessing

## Fichiers concernés

| Fichier | Rôle |
|---|---|
| `src/data.py` | Fonctions de chargement, nettoyage, feature engineering, encodage |
| `src/metrics.py` | Calcul et comparaison des métriques d'évaluation |
| `notebooks/preprocessing.ipynb` | Pipeline complet exécutable, produit les CSV |

## Données / Notebooks

### Comment les datasets transformés ont été obtenus

En exécutant `notebooks/preprocessing.ipynb` (Run All). Le notebook applique dans l'ordre : nettoyage → feature engineering → label encoding → split → OneHotEncoder → StandardScaler.

### Où ils sont stockés

```
data/
  raw/
    student-mat.csv          ← données brutes Mathématiques
    student-por.csv          ← données brutes Portugais
  processed/
    student_train.csv        ← X_train scalé + y_train  (Régression Logistique)
    student_test.csv         ← X_test scalé  + y_test
    student_train_raw.csv    ← X_train non scalé + y_train  (RF, Gradient Boosting)
    student_test_raw.csv     ← X_test non scalé  + y_test
```

### Comment les charger et les utiliser

```python
import pandas as pd
from pathlib import Path

DATA = Path('data/processed')

# Pour la Régression Logistique (données scalées)
train = pd.read_csv(DATA / 'student_train.csv')
test  = pd.read_csv(DATA / 'student_test.csv')
X_train, y_train = train.drop(columns=['pass']), train['pass']
X_test,  y_test  = test.drop(columns=['pass']),  test['pass']

# Pour RF et Gradient Boosting (données non scalées)
train_r = pd.read_csv(DATA / 'student_train_raw.csv')
test_r  = pd.read_csv(DATA / 'student_test_raw.csv')
X_train_r = train_r.drop(columns=['pass'])
X_test_r  = test_r.drop(columns=['pass'])
```

---

## 1. Étapes de nettoyage des données

Le dataset Student Performance ne contient aucune valeur manquante ni doublon. Les opérations de nettoyage sont toutes **déterministes** (seuils codés en dur) — applicables avant le split sans data leakage.

### 1.1 Valeurs manquantes
- **Résultat** : 0 valeur manquante sur 1044 lignes
- **Action** : aucune imputation nécessaire

### 1.2 Doublons
- **Résultat** : 0 doublon strict
- **Action** : aucune suppression

### 1.3 Capping des absences à 30
La variable `absences` est très asymétrique (max=75, p99≈30, médiane≈4). On plafonne à 30 — seuil fixe, non calculé depuis les données, donc sans leakage.

### 1.4 Suppression de G1, G2, G3
- **G1 et G2** : notes intermédiaires non disponibles en début d'année — les inclure viderait l'objectif de détection précoce
- **G3** : source de la variable cible `pass`, sa présence constituerait un leakage direct

---

## 2. Transformations appliquées

### 2.1 Label encoding des variables binaires — avant le split

Mappings fixes (rien appris depuis les données) → applicables avant le split.

| Colonne | Mapping |
|---|---|
| `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic` | `yes` → 1, `no` → 0 |
| `sex` | `F` → 1, `M` → 0 |
| `address` | `U` → 1, `R` → 0 |
| `famsize` | `GT3` → 1, `LE3` → 0 |
| `Pstatus` | `T` → 1, `A` → 0 |
| `school` | `GP` → 1, `MS` → 0 |

**Justification** : mapping direct, aucune perte d'information, aucun paramètre appris.

### 2.2 One-Hot Encoding des variables nominales — après le split, fit sur train uniquement

Les colonnes nominales multi-classes (`Mjob`, `Fjob`, `reason`, `guardian`, `course`) sont encodées via `OneHotEncoder(handle_unknown='ignore')`, fitté uniquement sur `X_train`.

**Justification** : le OHE apprend le vocabulaire des catégories — le fitter sur le dataset entier ferait fuiter des informations du test set. `handle_unknown='ignore'` garantit qu'une catégorie inconnue dans le test produit une ligne nulle sans erreur.

**Pourquoi pas `pd.get_dummies`** : il ne mémorise pas les catégories vues au fit — risque de colonnes divergentes entre train et test.

### 2.3 StandardScaler — après le split, fit sur train uniquement

Toutes les features numériques sont centrées-réduites (µ=0, σ=1). Le scaler est fitté uniquement sur `X_train`, puis appliqué sur train et test séparément.

**Justification** : requis pour la Régression Logistique (convergence du gradient). Les modèles arborescents sont insensibles au scaling — deux versions des données sont produites (scalée et non scalée).

---

## 3. Nouvelles features créées

Toutes les nouvelles features sont des **fonctions déterministes** des colonnes existantes. Aucune statistique n'est apprise — elles peuvent être créées avant le split sans leakage.

| Feature | Construction | Justification |
|---|---|---|
| `alc_total` | `Dalc + Walc` | `Dalc` et `Walc` corrélés (r > 0.6) — somme réduit la redondance, signal alcool plus stable |
| `alc_high_risk` | `alc_total ≥ 5` | Seuil fixe — distingue consommation modérée vs à risque (r = -0.18 avec pass) |
| `parent_edu` | `(Medu + Fedu) / 2` | `Medu` et `Fedu` corrélés (r ≈ 0.6) — moyenne capture le capital éducatif familial sans multicolinéarité |
| `study_vs_social` | `studytime - goout` | Arbitrage étude / vie sociale — signal composite plus fort que chaque variable seule |
| `motivated_with_resources` | `higher==yes AND internet==yes` | Motivation + accès aux ressources — les deux ensemble sont plus prédictifs que séparément |
| `family_capital` | `parent_edu × famrel` | Parents éduqués dans un bon contexte familial — effet multiplicatif |
| `has_support` | `schoolsup==yes OR famsup==yes` | Au moins une source de soutien — OR car les deux sont substituables |
| `digital_access` | `address==U AND internet==yes` | Bonnes conditions de travail à domicile |
| `risk_score` | combinaison pondérée | Score composite des principaux facteurs d'échec identifiés dans l'EDA |

**Formule du `risk_score`** :
```
risk_score = failures × 2
           + alc_high_risk
           + (absences > 10)
           + (studytime == 1)
           - (higher == yes)
clipé à 0 minimum
```
`failures` est pondéré ×2 car c'est le prédicteur le plus fort (r = -0.37 avec pass).

**Colonnes supprimées après FE** : `Dalc`, `Walc`, `Medu`, `Fedu` (remplacées par leurs agrégats).

---

## 4. Alternatives testées et non retenues

| Transformation | Résultat | Raison du rejet |
|---|---|---|
| `avg_grade_12 = (G1 + G2) / 2` | Amélioration massive | **Data leakage** — G1/G2 non disponibles en début d'année |
| LabelEncoder pour les nominales multi-classes | Performances comparables | Introduit un ordre artificiel entre catégories qui n'existe pas — OHE plus correct |
| MinMaxScaler | Performances comparables | Sensible aux outliers (`absences` jusqu'à 75 avant capping) |
| RobustScaler | Différence < 0.3% sur F1 | StandardScaler suffit après capping — complexité inutile |
| PCA (95% variance) | ΔF1 = -0.017 | Nécessite 28/40 composantes (réduction 24%) — perte d'interprétabilité sans gain |
| `age²` | Amélioration < 0.1% | Corrélation faible de l'âge avec pass (r = -0.12) |

---

## 5. Justification des choix effectués

### Pourquoi le split avant l'OHE et le scaler ?

Le OHE et le StandardScaler **apprennent depuis les données**. Les fitter sur le dataset entier avant le split fait fuiter des informations du test set dans le pipeline — data leakage qui surestime les performances. Les transformations déterministes (capping, sommes, mappings fixes) n'apprennent rien et peuvent être appliquées avant le split.

### Pourquoi deux versions des données ?

La Régression Logistique requiert des données scalées. Random Forest et Gradient Boosting sont arborescents — insensibles à l'échelle. On produit une version scalée et une non scalée ; chaque modèle consomme la version adaptée.

---

## 6. Impact attendu des transformations sur les modèles

| Transformation | Régression Logistique | Random Forest / Gradient Boosting |
|---|---|---|
| StandardScaler | Indispensable — convergence du gradient | Aucun effet |
| OHE des nominales | Indispensable | Indispensable |
| `risk_score` | Fort — linéarise une combinaison de facteurs d'échec | Modéré — les arbres trouvent ces combinaisons seuls, mais `risk_score` les guide |
| `alc_total` | Modéré — réduit la multicolinéarité Dalc/Walc | Faible — les arbres gèrent la redondance |
| `study_vs_social` | Modéré — interaction non modélisable nativement par LR | Faible — les arbres capturent l'interaction seuls |
| Capping absences | Faible — réduit l'influence des outliers sur le scaler | Très faible — les arbres sont insensibles aux outliers |
| Suppression G1/G2 | Forte dégradation volontaire — prédiction honnête en début d'année | Idem |