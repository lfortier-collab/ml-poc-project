# Assignment 1 — Sujet ML

## Description du sujet
Mon projet porte sur la prédiction de conversion client. L’objectif est de prédire si un utilisateur va effectuer un achat ou non à partir de données liées à son comportement sur un site ou une application.
C’est un problème de classification binaire : le modèle doit prédire deux classes possibles, achat ou non-achat.
Ce sujet est intéressant car il a une utilité business directe. Une entreprise peut utiliser ce type de modèle pour mieux cibler ses campagnes marketing, identifier les clients les plus susceptibles d’acheter et améliorer son taux de conversion.

## Problématique
Comment prédire la probabilité qu’un utilisateur réalise un achat à partir de ses données de navigation, de profil et d’interaction avec une plateforme ?

L’objectif serait de construire un modèle capable d’identifier les utilisateurs ayant une forte probabilité de conversion afin d’aider une entreprise à prioriser ses actions marketing.

## Dataset idéal
Le dataset idéal contiendrait plusieurs types de variables :

- des informations utilisateur : âge, genre, pays, ville ;
- des données de comportement : nombre de visites, temps passé sur le site, pages consultées, produits vus, panier abandonné ou non ;
- des données marketing : source de trafic, campagne publicitaire, canal d’acquisition ;
- des données historiques : nombre d’achats passés, montant moyen dépensé, ancienneté du client ;
- une variable cible : achat ou non-achat.

La variable cible serait donc binaire :
0 = l’utilisateur n’a pas acheté  
1 = l’utilisateur a acheté

## Description du dataset choisi
Le dataset retenu est le Online Shoppers Purchasing Intention Dataset, téléchargé via Kaggle, qui héberge une copie conforme du dataset UCI original. Il contient les données de comportement de navigation de sessions sur un site e-commerce réel, collectées sur une période d'un an entre 2017 et 2018. Chaque ligne représente une session utilisateur unique, sans qu'un même utilisateur n'apparaisse deux fois.
Le dataset contient 12 330 sessions et 17 features originales. La variable cible Revenue indique si la session s'est terminée par un achat. La distribution est fortement déséquilibrée : 84,5 % des sessions ne se terminent pas par un achat contre 15,5 % qui convertissent, ce qui est réaliste mais devra être traité.
Les features couvrent le comportement de navigation (pages visitées, temps passé), des métriques d'engagement Google Analytics comme le taux de rebond, le taux de sortie et la valeur des pages (PageValues, la feature la plus corrélée à l'achat avec 0,49), ainsi que des variables contextuelles comme le mois, le type de visiteur ou le week-end.

## Méthode de collecte
Le dataset a été téléchargé depuis Kaggle (https://www.kaggle.com/datasets/henrysue/online-shoppers-intention), copie du dataset UCI publié sous licence CC BY 4.0. Référence : Sakar, C. & Kastro, Y. (2018), https://doi.org/10.24432/C5F88Q.
En complément, j'ai développé un script de scraping pour enrichir le dataset avec des features contextuelles externes : dates du Black Friday, Cyber Monday et Cyber Week calculées algorithmiquement, jours fériés américains via la librairie holidays, et dates historiques d'Amazon Prime Day. Ces données sont jointes sur la colonne Month et ajoutent 10 nouvelles features au dataset final.

## Justification du choix
Ce dataset correspond directement à ma problématique, sa qualité est garantie par une publication académique (Neural Computing & Applications, 2019),les données sont sur une année complète. Il pose aussi de vrais défis ML : déséquilibre de classes, feature dominante, et saisonnalité marquée qui justifie l'enrichissement par scraping.
Ce dernier point m'a amené à investiguer l'origine du site. Les corrélations montrent que les jours fériés américains expliquent bien les achats (0,149) alors que les soldes françaises sont quasi nulles (-0,024). Le pic de novembre correspond clairement au Black Friday. Le site est donc probablement américain ou international, ce qui a orienté tout l'enrichissement vers des événements commerciaux US.

## Objectif business et ML
L'objectif business est d'identifier en cours de session les visiteurs les plus susceptibles d'acheter, pour déclencher au bon moment une offre personnalisée ou prioriser le service client. L'enjeu est d'augmenter le taux de conversion sans augmenter les coûts marketing.
Du côté ML, il s'agit d'un problème de classification binaire supervisée.

## Métrique d'évaluation envisagée
L'accuracy est à exclure : un modèle prédisant systématiquement "pas d'achat" atteindrait 84,5 % sans rien apprendre. La métrique principale retenue est le F1-score sur la classe positive, complété par l'AUC-ROC pour la capacité discriminante globale. Le recall et la précision seront surveillés selon le coût métier des erreurs.