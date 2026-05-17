# Assignment 4 — Visualisations

## Notebooks

| Visualisation | Notebook | Section |
|---|---|---|
| Données brutes | `notebooks/exploration_data.ipynb` | Section 3.2 — Distribution de la target |
| Données après feature engineering | `notebooks/preprocessing.ipynb` | Section 2 — EDA (heatmap corrélations) |
| Performances des modèles | `notebooks/modelling.ipynb` | Section 7 — Comparaison des 3 modèles |

### Comment générer les visualisations

```bash
# 1. Exécuter exploration_data.ipynb (Run All)   → viz données brutes
# 2. Exécuter preprocessing.ipynb (Run All)       → viz données après FE
# 3. Exécuter modelling.ipynb (Run All)          → viz performances
```

---

## Visualisation 1 — Données brutes

**Notebook** : `notebooks/exploration_data.ipynb` — Section 3.2  
**Figure** : graphique 3 panneaux — distribution G3 / répartition des classes / taux de réussite par matière

### Objectif

Donner une première lecture de la variable cible avant toute transformation : comprendre comment les notes finales G3 se distribuent, quel est le déséquilibre entre les classes, et si ce déséquilibre varie selon la matière.

### Choix du type de graphique

- **Histogramme superposé** (panneau gauche) : adapté pour comparer la distribution d'une variable continue (G3) entre deux groupes (math / portugais) sur le même axe. La ligne verticale à 10 matérialise le seuil de décision qui définit la variable cible.
- **Camembert** (panneau central) : la question est simple (quelle proportion réussit / échoue ?) — deux parts suffisent et le camembert communique ce ratio immédiatement, même pour une audience non-technique.
- **Barres** (panneau droit) : deux catégories ordinales (math / portugais) comparées sur un taux — le graphique en barres avec annotations de pourcentage est le plus lisible.

### Interprétation des résultats

La distribution de G3 révèle une bimodalité : un pic à 0 (élèves qui abandonnent ou échouent complètement) et un pic autour de 12-13. Le seuil à 10 crée un déséquilibre de classes notable : environ 78% de réussite contre 22% d'échec. Ce déséquilibre est similaire entre les deux matières, ce qui justifie de les concaténer. C'est cette observation qui impose le choix du F1-score macro comme métrique principale plutôt que l'accuracy.

### Pertinence pour le projet

Cette visualisation est le point de départ du projet : elle justifie à la fois le choix de la métrique (F1-macro face au déséquilibre) et la nécessité de concaténer les deux datasets (comportement similaire entre matières).

---

## Visualisation 2 — Données après Feature Engineering

**Notebook** : `notebooks/preprocessing.ipynb` — Section 2 (EDA)  
**Figure** : heatmap de corrélations absolues entre les features numériques

### Objectif

Identifier les redondances entre features avant de décider quelles variables fusionner ou supprimer. Valider que les choix de feature engineering (fusion `Dalc + Walc`, fusion `Medu + Fedu`) sont fondés sur les données.

### Choix du type de graphique

La **heatmap de corrélations absolues** est le graphique standard pour visualiser les relations entre toutes les paires de variables numériques simultanément. La valeur absolue est choisie (plutôt que la corrélation signée) pour mettre en évidence les redondances indépendamment du sens — ce qui est pertinent pour détecter la multicolinéarité. La diagonale est masquée pour éviter la distraction visuelle de la corrélation d'une variable avec elle-même.

### Interprétation des résultats

La heatmap révèle clairement :
- `Dalc` et `Walc` sont corrélés à r > 0.6 → fusion en `alc_total` justifiée
- `Medu` et `Fedu` sont corrélés à r ≈ 0.6 → fusion en `parent_edu` justifiée
- `studytime` et `goout` sont anti-corrélés → interaction `study_vs_social = studytime - goout` pertinente
- `failures` est relativement indépendant des autres features → conservé tel quel, pondéré ×2 dans `risk_score`

### Pertinence pour le projet

Cette visualisation est la justification empirique du feature engineering. Sans elle, les fusions de variables pourraient sembler arbitraires. Elle démontre que les choix effectués (agrégats, suppressions de colonnes redondantes) réduisent la multicolinéarité et produisent un espace de features plus propre pour les modèles.

---

## Visualisation 3 — Performances des modèles

**Notebook** : `notebooks/modelling.ipynb` — Section 7  
**Figure** : comparaison des métriques (barres groupées) + matrices de confusion (3 modèles) + courbes ROC

### Objectif

Comparer les 3 modèles sous plusieurs angles complémentaires : performance globale sur plusieurs métriques, capacité à discriminer sur tout le spectre de seuils (ROC), et analyse détaillée des erreurs par type (matrice de confusion).

### Choix du type de graphique

- **Barres groupées** : permet de lire d'un coup les 4 métriques (F1 macro, F1 Échec, F1 Réussite, Accuracy) pour les 5 modèles (2 baselines + 3 modèles). Les annotations sur chaque barre évitent d'avoir à extraire les valeurs d'un tableau séparé.
- **Matrices de confusion** (3 panneaux côte à côte) : détaillent les 4 types d'erreurs pour chaque modèle. C'est indispensable en contexte éducatif — un Faux Négatif (élève en difficulté prédit comme réussissant) est beaucoup plus problématique qu'un Faux Positif, et la matrice permet de quantifier chaque type d'erreur séparément.
- **Courbes ROC** : évaluent la performance indépendamment du seuil de décision. L'AUC résume la courbe en un chiffre comparable entre modèles. C'est particulièrement utile avec des classes déséquilibrées car la courbe ROC ne dépend pas du ratio de classes.

### Interprétation des résultats

Les barres groupées montrent que les 3 modèles dépassent clairement les 2 baselines (Dummy), ce qui confirme qu'ils apportent une valeur réelle. Le Gradient Boosting obtient le meilleur F1 macro, suivi par le Random Forest puis la Régression Logistique. L'accuracy (~80%) est trompeuse — le F1 Échec (classe minoritaire) est nettement plus faible, ce qui confirme que l'accuracy seule n'est pas une bonne métrique ici. Les matrices de confusion révèlent que tous les modèles ont tendance à sous-prédire la classe Échec — ce sont les Faux Négatifs qui restent le principal point d'amélioration.

### Pertinence pour le projet

Cette visualisation conclut le projet en permettant de prendre une décision éclairée : si la performance brute prime, on choisit le Gradient Boosting ; si l'interprétabilité est requise (directeur d'école devant expliquer les prédictions), la Régression Logistique reste défendable avec un écart de performance modéré. La visualisation multi-angle empêche de se fier à une seule métrique et force une analyse honnête des limites de chaque modèle.

---

## Justification globale

Les trois visualisations couvrent les trois étapes du cycle ML et forment une narration cohérente :

1. **Données brutes** → comprendre le problème, justifier la métrique et la concaténation des datasets
2. **Après feature engineering** → valider empiriquement que les transformations réduisent la redondance
3. **Performances des modèles** → comparer les modèles de façon honnête et multi-dimensionnelle, identifier les erreurs critiques

