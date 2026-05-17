# Assignment 3 — Modélisation

## Notebooks

| Notebook | Localisation | Rôle |
|---|---|---|
| `preprocessing.ipynb` | `notebooks/preprocessing.ipynb` | Nettoyage, feature engineering, split, OHE, scaling → produit les CSV |
| `modelling.ipynb` | `notebooks/modelling.ipynb` | Chargement des CSV, entraînement, tuning, évaluation des 3 modèles |

### Comment reproduire les expériences

```bash
# 1. Depuis le dossier notebooks/
jupyter notebook

# 2. Exécuter preprocessing.ipynb (Run All)
#    → produit data/processed/student_train.csv, student_test.csv,
#               student_train_raw.csv, student_test_raw.csv

# 3. Exécuter modelling.ipynb (Run All)
#    → produit models/*.joblib et results/model_metrics.csv
```

---

## 1. Définition du problème ML

**Type** : Classification binaire supervisée

**Variable cible** : `pass`
- `1` (Réussite) si la note finale G3 ≥ 10
- `0` (Échec) si G3 < 10

**Contexte** : prédire la réussite d'un élève **en début d'année**, à partir de son profil personnel, familial et comportemental uniquement. Les notes intermédiaires G1 et G2 sont volontairement exclues — les utiliser rendrait la prédiction triviale (corrélation > 0.8 avec G3) et ne permettrait pas une détection précoce.

**Objectif business** : identifier les élèves à risque d'échec dès le début de l'année scolaire pour permettre une intervention pédagogique précoce.

---

## 2. Métrique d'évaluation

**Métrique principale : F1-Score macro**

Le dataset présente un déséquilibre de classes (78% réussite / 22% échec). Le F1-Score macro calcule la moyenne harmonique précision/rappel séparément pour chaque classe, puis en fait la moyenne — il accorde ainsi le même poids aux deux classes, indépendamment de leur fréquence.

Concrètement, un modèle qui prédirait "réussite" pour tout le monde obtiendrait ~78% d'accuracy mais un F1 macro de ~0.44 — la métrique pénalise bien l'ignorance de la classe minoritaire, ce qui est exactement le cas d'usage critique ici : **rater un élève en difficulté est plus coûteux que signaler un faux positif**.

**Métriques secondaires** :
- F1-Score par classe (Échec et Réussite séparément) — pour analyser les erreurs
- AUC-ROC — robustesse au seuil de décision
- Accuracy — lisibilité pour les parties prenantes non-techniques

---

## 3. Protocole d'évaluation

### Split train / test
- **80% train / 20% test**, stratifié sur la cible (`stratify=y`)
- La stratification garantit que le ratio 78%/22% est respecté dans les deux ensembles
- `random_state=42` pour la reproductibilité
- **Le test set n'est jamais touché avant l'évaluation finale** — aucun hyperparamètre n'est ajusté en le regardant

### Cross-validation
- **5-fold stratifiée** sur le train uniquement
- Utilisée pour comparer les modèles et pour le GridSearchCV
- La stratification par fold garantit la représentation de la classe minoritaire dans chaque fold

### Sélection d'hyperparamètres
- **GridSearchCV** sur 5-fold CV, scoring = F1-macro
- `refit=True` : le modèle est refitté sur tout le train avec les meilleurs paramètres trouvés

### Évaluation finale
- Chaque modèle est évalué **une seule fois** sur le test set, après la sélection des hyperparamètres

---

## 4. Présentation des trois modèles

### Modèle 1 — Régression Logistique

**Hypothèses principales**
La régression logistique suppose qu'il existe une relation **linéaire** entre les features (après scaling) et le log-odds de la variable cible. Elle suppose également que les features sont relativement indépendantes entre elles.

**Avantages attendus**
- Entièrement interprétable : les coefficients indiquent directement l'influence de chaque feature
- Rapide à entraîner, peu de risque d'overfitting sur un dataset de cette taille
- Bon baseline — si un modèle plus complexe ne fait pas mieux, la complexité n'est pas justifiée

**Limites attendues**
- Incapable de capturer des interactions non-linéaires (ex : l'effet de `failures` peut être non-linéaire)
- Performances plafonnées si les relations dans les données sont effectivement non-linéaires
- Sensible au scaling — requiert des données normalisées (StandardScaler appliqué)

**Adéquation avec le problème et la métrique**
En contexte éducatif, l'interprétabilité est critique : un directeur d'école doit pouvoir comprendre pourquoi un élève est signalé à risque. La régression logistique remplit ce rôle. Sur le F1-macro, elle sera compétitive si les features engineered (`risk_score`, `study_vs_social`) ont bien linéarisé les relations clés.

---

### Modèle 2 — Random Forest

**Hypothèses principales**
Un ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires du dataset et des sous-ensembles de features peut approcher des fonctions complexes tout en limitant l'overfitting par la loi des grands nombres (bagging).

**Avantages attendus**
- Capture naturellement les non-linéarités et les interactions entre features
- Robuste aux outliers (le capping des absences à 30 est une précaution, mais l'arbre le gèrerait de toute façon)
- Feature importance native — utile pour valider le feature engineering
- Peu sensible aux hyperparamètres par défaut
- Fonctionne sur données non scalées

**Limites attendues**
- Moins interprétable qu'un modèle linéaire (modèle "boîte noire")
- Peut légèrement over-fitter si les arbres sont trop profonds — contrôlé par `max_depth` dans le GridSearchCV

**Adéquation avec le problème et la métrique**
Le Random Forest est un excellent choix pour des données tabulaires mixtes (numériques + catégorielles encodées). Il devrait bien gérer le déséquilibre de classes et maximiser le F1-macro en apprenant des règles de décision complexes que la régression logistique ne peut pas exprimer.

---

### Modèle 3 — Gradient Boosting

**Hypothèses principales**
Un ensemble d'arbres peu profonds construits **séquentiellement** — chaque arbre corrige les erreurs du précédent en se concentrant sur les observations mal classées. Le modèle apprend ainsi à minimiser une fonction de coût de façon itérative.

**Avantages attendus**
- Généralement les meilleures performances sur des datasets tabulaires de petite à moyenne taille
- Gère naturellement les interactions complexes entre features
- Régularisation intégrée (`subsample`, `learning_rate`, `max_depth`) qui limite l'overfitting
- Fonctionne sur données non scalées

**Limites attendues**
- Plus sensible aux hyperparamètres que le Random Forest — le GridSearchCV est indispensable
- Plus long à entraîner (nature séquentielle)
- Risque d'overfitting si `learning_rate` trop élevé ou `n_estimators` trop grand sans régularisation

**Adéquation avec le problème et la métrique**
Sur un dataset de 1044 lignes avec un signal diffus (les features comportementales ont des corrélations modérées avec la cible), le Gradient Boosting est généralement le modèle le plus performant. Il devrait maximiser le F1-macro en combinant des dizaines de règles de décision faibles en un prédicteur fort.

---

## 5. Justification du choix des trois modèles

Les trois modèles ont été choisis pour couvrir trois niveaux de complexité complémentaires :

**Diversité des hypothèses** : la régression logistique suppose la linéarité, le Random Forest suppose que des règles locales indépendantes suffisent, le Gradient Boosting suppose qu'un apprentissage séquentiel corrigeant les erreurs est plus efficace. Tester les trois permet de déterminer quelle hypothèse correspond le mieux aux données.

**Compromis interprétabilité / performance** : en contexte éducatif, l'interprétabilité a une valeur réelle. Si la régression logistique atteint une performance proche du Gradient Boosting (écart < 2 points de F1-macro), elle sera préférée pour sa lisibilité. Les trois modèles permettent de prendre cette décision en connaissance de cause.

**Référence industrielle** : ces trois modèles constituent la référence standard pour la classification sur données tabulaires. Leur inclusion permet de situer les résultats dans un contexte connu et reproductible.