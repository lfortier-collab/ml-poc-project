# Assignment 1 — Définition du projet et exploration des données

## Description du projet

Ce projet vise à prédire la **réussite scolaire d'un étudiant** à partir de ses caractéristiques personnelles, familiales et comportementales. L'objectif est d'identifier les élèves à risque d'échec avant les résultats finaux, afin de permettre une intervention pédagogique précoce.

---

## Définition du problème

**Type de problème : Classification binaire**

- **Variable cible** : `pass` — créée à partir de la note finale G3
  - `1` (Réussite) si G3 ≥ 10
  - `0` (Échec) si G3 < 10
- **Entrées** : 30 features socio-démographiques, comportementales et scolaires
- **Approche** : apprentissage supervisé

---

## Description du dataset

| Attribut | Valeur |
|---|---|
| **Source** | UCI Machine Learning Repository — Student Performance Dataset |
| **Auteur** | Paulo Cortez, Université du Minho (Portugal) |
| **Licence** | Open / usage académique libre |
| **Lien** | https://archive.ics.uci.edu/ml/datasets/Student+Performance |
| **Fichiers** | `studentmat.csv` (395 élèves, Mathématiques) + `studentpor.csv` (649 élèves, Portugais) |
| **Total** | 1044 lignes après concaténation |
| **Valeurs manquantes** | Aucune |

### Comment le dataset a été obtenu

Téléchargé directement depuis l'UCI ML Repository. Les deux fichiers CSV sont stockés dans `data/raw/`.

### Localisation dans le repository

```
data/
  raw/
    studentmat.csv    ← dataset Mathématiques
    studentpor.csv    ← dataset Portugais
  processed/
    student_processed.csv  ← dataset après feature engineering
```

### Comment utiliser les données

```python
import pandas as pd
df_mat = pd.read_csv("data/raw/studentmat.csv", sep=";")
df_por = pd.read_csv("data/raw/studentpor.csv", sep=";")
```

---

## Description des features disponibles

### Features démographiques
| Feature | Type | Description |
|---|---|---|
| `school` | Binaire | École : GP (Gabriel Pereira) ou MS (Mousinho da Silveira) |
| `sex` | Binaire | Sexe : F / M |
| `age` | Numérique | Âge (15 à 22 ans) |
| `address` | Binaire | Adresse : U (urbain) ou R (rural) |
| `famsize` | Binaire | Taille famille : LE3 (≤3) ou GT3 (>3) |
| `Pstatus` | Binaire | Statut parental : T (ensemble) ou A (séparés) |

### Features familiales
| Feature | Type | Description |
|---|---|---|
| `Medu` / `Fedu` | Numérique (0-4) | Niveau d'éducation de la mère / du père |
| `Mjob` / `Fjob` | Nominal | Profession de la mère / du père |
| `guardian` | Nominal | Tuteur légal : mère, père ou autre |
| `famrel` | Numérique (1-5) | Qualité des relations familiales |
| `famsup` | Binaire | Soutien éducatif familial |

### Features scolaires
| Feature | Type | Description |
|---|---|---|
| `reason` | Nominal | Raison du choix de l'école |
| `traveltime` | Numérique (1-4) | Temps de trajet domicile-école |
| `studytime` | Numérique (1-4) | Temps d'étude hebdomadaire |
| `failures` | Numérique | Nombre d'échecs passés |
| `schoolsup` | Binaire | Soutien scolaire supplémentaire |
| `paid` | Binaire | Cours particuliers payants |
| `absences` | Numérique | Nombre d'absences (0 à 93) |
| `G1` / `G2` | Numérique (0-20) | Notes 1ère et 2ème période |
| `G3` | Numérique (0-20) | Note finale (**non utilisée comme feature**) |

### Features comportementales
| Feature | Type | Description |
|---|---|---|
| `activities` | Binaire | Activités extra-scolaires |
| `nursery` | Binaire | Crèche fréquentée |
| `higher` | Binaire | Souhaite faire des études supérieures |
| `internet` | Binaire | Accès internet à domicile |
| `romantic` | Binaire | En couple |
| `freetime` | Numérique (1-5) | Temps libre après l'école |
| `goout` | Numérique (1-5) | Sorties avec les amis |
| `Dalc` / `Walc` | Numérique (1-5) | Consommation d'alcool semaine / weekend |
| `health` | Numérique (1-5) | État de santé actuel |

---

## Premières analyses exploratoires (EDA)

Le notebook EDA est disponible dans : `notebooks/eda.ipynb`

**Principales observations :**
- **78% de réussite** (G3 ≥ 10) — classes légèrement déséquilibrées
- `G1` et `G2` sont fortement corrélés à `G3` (signal prédictif majeur)
- Les élèves avec plus d'échecs passés (`failures`) ont un taux de réussite nettement inférieur
- Plus le temps d'étude est élevé, meilleur est le taux de réussite
- La consommation d'alcool élevée est corrélée négativement aux résultats

---

## Objectif Business

**Contexte** : Les établissements scolaires souhaitent réduire le taux d'échec en détectant tôt les élèves en difficulté.

**Objectif** : Construire un modèle capable de prédire, à partir des caractéristiques d'un élève (sans attendre la note finale G3), s'il est susceptible de réussir ou d'échouer. Cela permet de déclencher un accompagnement personnalisé dès le début de l'année.

---

## Contexte Machine Learning

| Élément | Choix |
|---|---|
| **Type de tâche** | Classification binaire supervisée |
| **Variable cible** | `pass` (0 = échec, 1 = réussite) |
| **Features utilisées** | 37 features (originales + engineered) |
| **Modèles envisagés** | Régression Logistique, Random Forest, Gradient Boosting |
| **Split** | 80% train / 20% test, stratifié |
| **Validation** | Cross-validation 5-fold stratifiée |

---

## Métrique et fonction de coût envisagée

**Métrique principale : F1-Score (macro)**

Justification : les classes sont déséquilibrées (78% réussite / 22% échec). Le F1-Score pénalise les modèles qui ignorent la classe minoritaire (les élèves en échec), ce qui est exactement le cas d'usage critique ici — rater un élève en difficulté est plus coûteux que signaler un faux positif.

**Métriques secondaires** :
- Accuracy (lisibilité)
- AUC-ROC (robustesse au seuil de décision)
- Matrice de confusion (analyse des erreurs)

---

## Hypothèses, risques et limites identifiées

### Hypothèses
- Un étudiant avec G3 ≥ 10 est considéré comme ayant réussi
- Les features G1 et G2 (notes intermédiaires) sont disponibles au moment de la prédiction
- Les données des deux matières (Math + Portugais) sont suffisamment homogènes pour être combinées

### Risques
- **Data leakage** : G1 et G2 sont des notes intermédiaires — elles peuvent ne pas être disponibles en début d'année. Une version du modèle sans G1/G2 pourrait être nécessaire pour une prédiction en amont.
- **Déséquilibre des classes** : 78% vs 22% — le modèle pourrait être biaisé vers la classe majoritaire.
- **Généralisation** : le dataset est issu de deux lycées portugais spécifiques, ce qui limite la généralisation à d'autres contextes géographiques ou culturels.

### Limites
- Taille du dataset modeste (1044 lignes) — peu propice aux modèles profonds
- Variables auto-déclarées (alcool, santé, relations) — potentiellement biaisées
- Pas d'information temporelle (évolution de l'élève dans le temps)
