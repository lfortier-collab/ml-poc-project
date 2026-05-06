# Assignment 2 — Preprocessing & Feature Engineering


## 1. Choix structurant : suppression de G1 et G2

Avant tout nettoyage, une analyse exploratoire a révélé que G1 et G2 sont corrélées à G3 avec des coefficients de **0.81 et 0.91**. Les inclure comme features poserait deux problèmes :

- **Data leakage fonctionnel** : l'objectif est de détecter les élèves en difficulté *en début d'année*, avant les résultats intermédiaires. G1 et G2 ne sont pas disponibles à ce stade.
- **Domination des features** : elles écraient toute l'information socio-démographique et comportementale, qui est au cœur du projet.

→ G1, G2 et G3 sont supprimées. La cible `pass` (1 si G3 ≥ 10) est créée dès le chargement.

---

## 2. Étapes de nettoyage des données

### 2.1 Valeurs manquantes
Aucune valeur manquante dans les deux datasets (`student-mat.csv` et `student-por.csv`). Aucune imputation nécessaire.

### 2.2 Doublons
Aucun doublon strict. Note : certains élèves apparaissent dans les deux datasets (math + portugais) — ce sont des observations légitimes pour deux cours différents, différenciées par la colonne `course`.

### 2.3 Outliers — `absences`
La variable `absences` présente des valeurs extrêmes (maximum : 75 jours, 99e percentile : ~30).

**Action** : capping à 30 jours.

**Justification** : seuls 6 élèves (< 1%) dépassent ce seuil. Ces valeurs extrêmes, si non traitées, biaisent le StandardScaler en dilatant l'échelle de la feature. On conserve l'information "très absent" sans laisser un cas isolé dicter l'échelle de toute la feature.

---

## 3. Feature Engineering

### 3.1 Observations qui ont motivé le feature engineering

L'analyse exploratoire préalable (corrélations, distributions, heatmap) a identifié :

| Observation | Feature créée |
|---|---|
| `failures` : meilleur prédicteur négatif (-0.37) | Pondéré ×2 dans `risk_score` |
| `Dalc` et `Walc` très corrélés (r > 0.6) | Fusionnés en `alc_total`, puis supprimés |
| `Medu` et `Fedu` corrélés (r ~0.6) | Fusionnés en `parent_edu`, puis supprimés |
| `studytime` positif, `goout` négatif | Interaction capturée par `study_vs_social` |
| `higher` fort prédicteur positif | Combiné dans `motivated_with_resources` + `risk_score` |

### 3.2 Nouvelles features créées

| Feature | Construction | Justification |
|---|---|---|
| `alc_total` | `Dalc + Walc` | Dalc et Walc mesurent le même comportement (alcool) sous deux angles corrélés. Un score unique réduit la redondance. |
| `alc_high_risk` | `alc_total >= 5` (binaire) | Seuil à 50% du max (10) — capte un risque comportemental élevé. |
| `parent_edu` | `(Medu + Fedu) / 2` | Capital éducatif moyen du foyer. Medu et Fedu corrélés → un seul score évite la multicolinéarité. |
| `study_vs_social` | `studytime - goout` | Capture l'arbitrage entre temps d'étude et vie sociale — dynamique absente des features brutes. |
| `motivated_with_resources` | `higher == 'yes' AND internet == 'yes'` | Motivation + ressources numériques : combinaison plus prédictive que les deux séparément. |
| `family_capital` | `parent_edu × famrel` | Des parents éduqués dans un foyer aux relations dégradées ont un impact limité. |
| `has_support` | `schoolsup == 'yes' OR famsup == 'yes'` | Présence d'au moins une source de soutien actif. |
| `digital_access` | `address == 'U' AND internet == 'yes'` | Élève urbain avec internet = meilleures ressources de travail à domicile. |
| `risk_score` | `failures×2 + alc_high_risk + (absences>10) + (studytime==1) - (higher=='yes')`, clippé à 0 | Score composite des facteurs d'échec les plus prédictifs. `failures` pondéré ×2 car c'est le signal le plus fort (-0.37). `higher` soustrait comme facteur protecteur. |

**Features supprimées après engineering** : `Dalc`, `Walc`, `Medu`, `Fedu` — remplacées par des features plus riches et moins redondantes.

---

## 4. Transformations appliquées

### 4.1 Encoding

| Type | Colonnes | Méthode |
|---|---|---|
| Binaires yes/no | `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic` | Label encoding (0/1) |
| Binaires 2 valeurs | `sex`, `address`, `famsize`, `Pstatus`, `school` | Label encoding (0/1) |
| Nominales multi-classes | `Mjob`, `Fjob`, `reason`, `guardian`, `course` | One-Hot Encoding |

### 4.2 Scaling

**StandardScaler** (µ=0, σ=1) appliqué sur l'ensemble des features.

**Quand utilisé** : obligatoire pour la PCA (sensible aux échelles), recommandé pour la régression logistique (convergence, régularisation). Inutile pour Random Forest et Gradient Boosting (splits insensibles aux échelles) mais le dataset scalé est fourni pour tous les modèles.

### 4.3 PCA

Une PCA complète a été appliquée et analysée (scree plot, variance cumulée, projection 2D).

**Résultat** : atteindre 90% de variance requiert 58% des dimensions originales — gain marginal. Les deux classes se chevauchent fortement dans l'espace PC1/PC2.

**Décision : PCA non retenue pour le modèle principal.**

---

## 5. Justification des choix

| Choix | Justification |
|---|---|
| Capping absences à 30 | Robustesse du scaling, préservation du signal "fort absentéisme" |
| Suppression G1/G2 | Data leakage fonctionnel + objectif de détection précoce |
| Fusion Dalc+Walc | Réduction multicolinéarité, information équivalente en un seul score |
| Fusion Medu+Fedu | Idem — capital éducatif parental = concept unique |
| Failures ×2 dans risk_score | Corrélation la plus forte avec la cible (-0.37) |
| One-Hot sur nominales | Aucun ordre entre catégories (Mjob, Fjob...) — ordinal encoding créerait un ordre artificiel |
| StandardScaler vs MinMaxScaler | Plus robuste aux asymétries résiduelles après capping |
| PCA non retenue | Gain limité, perte d'interprétabilité critique dans un contexte éducatif |

---

## 6. Alternatives testées et non retenues

| Alternative | Pourquoi non retenue |
|---|---|
| **Garder G1 et G2** | Data leakage fonctionnel — non disponibles en début d'année ; corrélation > 0.8 avec G3 écrase les autres features |
| **Ordinal encoding** sur Mjob, Fjob, reason, guardian | Introduit un ordre artificiel entre catégories sans relation ordinale ("teacher > health" n'a aucun sens) |
| **Target encoding** | Risque de fuite d'information sur un dataset de 1044 lignes. Nécessiterait une validation croisée interne pour éviter l'overfitting |
| **MinMaxScaler** | Sensible aux valeurs extrêmes résiduelles (absences), moins robuste que StandardScaler pour des distributions asymétriques |
| **PCA pour le modèle principal** | Gain de compression marginal (~58% des dims pour 90% de variance) ; classes non séparables dans l'espace réduit ; perte d'interprétabilité des features individuelles (essentielle pour identifier les facteurs de risque) |
| **Conserver Dalc et Walc séparément** | Corrélation r > 0.6 entre eux — redondance utile à réduire pour les modèles linéaires |
| **Log-transform des absences** | Capping à 30 suffit et préserve mieux l'interprétabilité |
| **parent_edu_gap** (écart Medu-Fedu) | Corrélation quasi-nulle avec la cible (r ≈ -0.01) — feature non informative |

---

## 7. Impact attendu des transformations sur les modèles

### 7.1 Nettoyage
- **Suppression G1/G2** : les modèles devront trouver le signal dans les features socio-démographiques et comportementales. Les performances seront moindres qu'avec G1/G2, mais le modèle sera réellement utile en pratique.
- **Capping absences** : évite que les valeurs extrêmes biaisent la régularisation des modèles linéaires et le scaling pour la PCA.

### 7.2 Feature Engineering
- **`risk_score`** : fournit un signal agrégé fort que les modèles linéaires peuvent exploiter directement. Pour les modèles arborescents, les features composantes individuelles sont déjà disponibles.
- **`study_vs_social`** : capture une interaction que les modèles linéaires ne peuvent pas construire eux-mêmes (interaction multiplicative/additive entre features).
- **`motivated_with_resources`** : interaction logique que ni Random Forest ni Gradient Boosting ne construisent nécessairement sans feature explicite.
- **Suppression Dalc/Walc/Medu/Fedu** : réduit la multicolinéarité, ce qui améliore la stabilité des coefficients en régression logistique.

### 7.3 Encoding & Scaling
- **One-Hot** : rend les nominales exploitables par tous les modèles, sans biais ordinal.
- **StandardScaler** : crucial pour la régression logistique (L1/L2 régularisation équitable entre features). Neutre pour RF/GBM.

### 7.4 PCA (si utilisée avec régression logistique)
- Élimine la multicolinéarité résiduelle entre features OHE.
- Coût : perte totale de l'interprétabilité — les coefficients du modèle ne correspondent plus à des features métier.

---

## 8. Datasets produits

  student_processed.csv   ← dataset encodé, non scalé (pour RF, GBM)
  student_scaled.csv      ← dataset encodé + scalé (pour Régression Logistique)
  student_pca.csv         ← dataset réduit PCA 90% variance (option)
```