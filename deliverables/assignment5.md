# Assignment 5 — Justification des choix techniques

---

## 1. Seuil de réussite : G3 ≥ 10

**Choix** : un élève est considéré comme ayant réussi si sa note finale G3 est supérieure ou égale à 10.

**Justification** :
- Le système scolaire portugais utilise une notation sur 20. La note de passage officielle est **10/20**, ce seuil reflète donc directement la réalité institutionnelle.
- Ce n'est pas un seuil arbitraire : c'est la frontière réglementaire qui détermine si un élève valide son année.
- La distribution de G3 présente une bimodalité autour de 0 (abandons) et 12-13 (réussites typiques), avec peu d'élèves entre 8 et 11 — le seuil à 10 coupe naturellement les deux populations.

---

## 2. Seuil de décision : 0.5

**Choix** : dans la page de démonstration, un élève est prédit "Réussite" si P(réussite) ≥ 0.5, "Échec" sinon.

**Justification** :
- Le seuil 0.5 est le seuil neutre : il minimise l'erreur globale quand les coûts des deux types d'erreur sont équivalents.
- **Pourquoi ne pas l'abaisser pour détecter plus d'élèves en échec ?** En production (intervention pédagogique réelle), abaisser le seuil augmente le rappel sur la classe "Échec" mais génère plus de faux positifs — des élèves qui réussissent et qui mobiliseraient inutilement des ressources. Le seuil optimal dépend du contexte métier et du coût relatif de chaque erreur.
- Le seuil 0.5 est retenu ici comme **valeur de référence interprétable** pour la démonstration. En production, il serait calibré sur les données de validation en maximisant le F1-score sur la classe "Échec".

---

## 3. Split train/test : 80 % / 20 %

**Choix** : 835 exemples en train, 209 en test (`random_state=42`, `stratify=y`).

**Justification** :
- Le dataset contient 1 044 lignes — un dataset de taille modeste. Avec un split 80/20, on conserve **835 exemples pour l'entraînement**, ce qui est suffisant pour entraîner les 3 modèles sans underfitting.
- Le test set de **209 exemples** est assez grand pour produire des métriques stables : avec 22 % de taux d'échec, on obtient ~46 vrais négatifs, ce qui permet d'estimer le F1 sur la classe minoritaire.
- La **stratification** préserve le ratio 78 %/22 % dans les deux splits — sans elle, un split aléatoire pourrait produire un test set avec 15 % ou 30 % d'échecs, rendant les métriques incomparables.
- `random_state=42` garantit la **reproductibilité** — deux exécutions du pipeline donnent exactement les mêmes splits.
- **Alternatives testées non retenues** : 70/30 aurait réduit le train set à 730 exemples (risque d'underfitting pour le GB), 90/10 aurait produit un test set de 104 exemples (~23 exemples "Échec"), trop petit pour estimer le F1 de façon fiable.

---

## 4. Validation croisée : 5-fold stratifiée

**Choix** : 5-fold cross-validation stratifiée sur le train set pour sélectionner les hyperparamètres.

**Justification** :
- Sur 835 exemples d'entraînement, **5 folds** donnent des folds de ~167 exemples (train interne : ~668, validation interne : ~167). C'est un équilibre entre variance des estimations (folds trop petits → estimations bruitées) et coût computationnel (folds trop nombreux → 10-fold ou LOOCV très lents pour le GB).
- La **stratification** est indispensable : sans elle, un fold pourrait contenir très peu d'exemples "Échec" (classe à 22 %), rendant le F1 macro instable.
- **5-fold vs 10-fold** : sur un dataset de cette taille, 5-fold et 10-fold donnent des estimations proches. 5-fold est retenu pour le coût computationnel réduit (5 entraînements vs 10 par configuration d'hyperparamètres).

---

## 5. Gestion du déséquilibre : `class_weight='balanced'`

**Choix** : `class_weight='balanced'` pour la Régression Logistique et le Random Forest.

**Justification** :
- Avec 78 % de réussite, un modèle non pondéré apprend rapidement à prédire "toujours Réussite" — il minimise la perte globale sans rien apprendre sur les élèves en échec.
- `class_weight='balanced'` multiplie le poids de chaque exemple par `n_samples / (n_classes × n_samples_per_class)`, soit environ **3.5× plus de poids** pour la classe "Échec". Cela force le modèle à ne pas ignorer la classe minoritaire.
- **Alternative non retenue — SMOTE** : le suréchantillonnage synthétique crée de nouveaux exemples artificiels. Sur un dataset de 1 044 lignes avec des variables mixtes (binaires OHE + ordinales), SMOTE génère des exemples peu réalistes. La pondération est plus simple et aussi efficace.
- **Gradient Boosting** : `class_weight` n'est pas supporté nativement par `GradientBoostingClassifier`. Le déséquilibre est compensé par l'optimisation itérative — le GB se concentre naturellement sur les exemples mal classifiés, qui sont souvent les exemples "Échec".

---

## 6. Hyperparamètres — Régression Logistique

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| `C` | 1.0 | Régularisation L2 standard. C=1.0 est un bon point de départ — ni trop régularisé (underfitting) ni trop peu (overfitting sur 835 exemples). |
| `max_iter` | 1000 | Le solver lbfgs converge rarement en moins de 100 itérations avec des données scalées mais dépasse les 100 par défaut sur des features corrélées. |
| `solver` | lbfgs (défaut) | Adapté aux datasets de taille modeste avec régularisation L2. |
| `class_weight` | balanced | Compense le déséquilibre 78/22 (voir section 5). |

---

## 7. Hyperparamètres — Random Forest

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 200 | 200 arbres offrent une variance faible sans coût computationnel excessif. En dessous de 100, la variance des prédictions est notable. Au-delà de 300, le gain est marginal. |
| `max_features` | sqrt (défaut) | Racine carrée du nombre de features par split — standard pour la classification. Évite que les features dominantes (risk_score, failures) soient toujours choisies. |
| `max_depth` | None (défaut) | Les arbres grandissent jusqu'à pureté. L'agrégation de 200 arbres corrige l'overfitting individuel sans limiter la profondeur. |
| `class_weight` | balanced | Compense le déséquilibre 78/22. |
| `random_state` | 42 | Reproductibilité. |

---

## 8. Hyperparamètres — Gradient Boosting

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 200 | 200 arbres avec un faible learning rate convergent vers une bonne solution sans overfitting sur 835 exemples. |
| `learning_rate` | 0.05 | Faible learning rate + beaucoup d'estimateurs : approche classique qui généralise mieux que `lr=0.1` + 100 estimateurs. Chaque arbre corrige une petite fraction de l'erreur résiduelle. |
| `max_depth` | 3 | Arbres peu profonds = apprenants faibles. C'est le principe du boosting : des modèles simples combinés séquentiellement. max_depth=3 évite l'overfitting sur un dataset de 1 044 lignes. |
| `subsample` | 0.8 | Stochastic Gradient Boosting : chaque arbre est entraîné sur 80 % des données (tirage aléatoire sans remise). Réduit la variance et améliore la généralisation. |
| `random_state` | 42 | Reproductibilité. |

---

## 9. Standardisation : StandardScaler

**Choix** : StandardScaler (µ=0, σ=1) appliqué sur X_train, puis transformé sur X_test.

**Justification** :
- La régression logistique avec régularisation L2 est **sensible aux échelles** : une feature avec une grande variance domine la pénalité de régularisation. Le StandardScaler normalise cette influence.
- Le scaler est **fit uniquement sur X_train** et appliqué à X_test — pas de data leakage. Fitter le scaler sur l'ensemble du dataset ferait "voir" au modèle des informations du test set pendant l'entraînement.
- **StandardScaler vs MinMaxScaler** : StandardScaler est plus robuste aux distributions asymétriques résiduelles (ex : `absences` après capping à 30). MinMaxScaler est sensible aux valeurs extrêmes.
- Random Forest et Gradient Boosting sont **insensibles au scaling** (les splits des arbres sont basés sur des rangs), mais les données scalées leur sont fournies pour uniformiser le pipeline.

---

## 10. Capping des absences : 30 jours

**Choix** : les valeurs d'`absences` supérieures à 30 sont ramenées à 30.

**Justification** :
- Le maximum observé est 75 jours, mais le 99e percentile est ~30. Seuls **6 élèves (< 1 %)** dépassent 30 absences.
- Sans capping, ces 6 valeurs extrêmes dilatent l'échelle de la feature après standardisation, réduisant la résolution pour 99 % des élèves.
- Le seuil 30 est choisi au **99e percentile** : on conserve l'information "très absent" sans laisser un cas isolé distordre l'échelle.

---

## 11. Suppression de G1 et G2

**Choix** : les notes intermédiaires G1 et G2 sont exclues du modèle.

**Justification** :
- G1 et G2 sont corrélées à G3 avec des coefficients de **0.81 et 0.91** — les inclure ferait dominer ces deux variables et rendrait la prédiction quasi-triviale.
- **Objectif business** : détecter les élèves en difficulté *en début d'année*, avant toute évaluation. G1 et G2 ne sont pas disponibles à ce moment. Un modèle entraîné avec G1/G2 serait inutilisable en pratique.
- Sans G1/G2, le modèle doit trouver le signal dans les variables socio-démographiques et comportementales — c'est précisément ce qui le rend actionnable dès la rentrée.

---

## 12. Choix de la métrique principale : F1-score macro

**Choix** : F1-score macro comme métrique d'optimisation et de comparaison.

**Justification** :
- Les classes sont déséquilibrées : **78 % réussite / 22 % échec**. Un modèle prédisant toujours "Réussite" obtiendrait 78 % d'accuracy sans rien apprendre.
- Le **F1 macro** calcule le F1 indépendamment pour chaque classe, puis fait la moyenne — il accorde le même poids à la classe "Échec" (22 %) et à la classe "Réussite" (78 %).
- Il combine **précision** (éviter les fausses alertes) et **rappel** (ne pas rater les élèves en difficulté), ce qui est adapté quand les deux types d'erreur ont un coût.
- **F1 pondéré** non retenu : le F1 pondéré par la fréquence des classes avantagerait la classe majoritaire "Réussite" et masquerait les performances sur "Échec".
