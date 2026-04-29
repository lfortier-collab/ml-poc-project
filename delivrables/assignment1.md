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