# 📝 100 Questions QCU - Machine Learning

**Examen Machine Learning Appliqué - 4ERP-BI & 4ARCTIC**  
**Module : MLA | Durée : 1h | Documents : NON Autorisé | Internet : NON Autorisé**

---

## 📋 Table des Matières
- [Questions 1-10 : CRISP-DM](#questions-1-10--crisp-dm)
- [Questions 11-20 : Data Preprocessing](#questions-11-20--data-preprocessing)
- [Questions 21-30 : Data Preprocessing (Suite)](#questions-21-30--data-preprocessing-suite)
- [Questions 31-40 : ACP/PCA](#questions-31-40--acppca)
- [Questions 41-50 : ACP/PCA (Suite)](#questions-41-50--acppca-suite)
- [Questions 51-60 : K-Means](#questions-51-60--k-means)
- [Questions 61-70 : DBSCAN](#questions-61-70--dbscan)
- [Questions 71-80 : KNN & SVM](#questions-71-80--knn--svm)
- [Questions 81-90 : Concepts Transversaux](#questions-81-90--concepts-transversaux)
- [Questions 91-100 : Cas Pratiques](#questions-91-100--cas-pratiques)

---

## Questions 1-10 : CRISP-DM

### Question 1
Quelle affirmation décrit le mieux la logique générale de la méthodologie CRISP-DM ?
- A. Un processus linéaire de la compréhension métier jusqu'au déploiement.
- B. Un cycle itératif favorisant les retours entre les différentes phases du projet.
- C. Une méthode strictement séquentielle, sans révision des étapes précédentes.
- D. Une approche exclusivement technique centrée sur la modélisation des données.

**✓ Réponse : B**

---

### Question 2
Quelle est la différence fondamentale entre la programmation traditionnelle et le machine Learning ?
- A. En programmation, les règles sont apprises automatiquement à partir des données.
- B. En machine Learning, les règles sont définies manuellement par le programmeur.
- C. En machine Learning, le système apprend les règles à partir des données plutôt que de les coder explicitement.
- D. Aucune différence : les deux reposent sur des instructions programmées.

**✓ Réponse : C**

---

### Question 3
Quelle stratégie peut aider à réduire le sur-apprentissage (overfitting) ?
- A. Augmenter la complexité du modèle.
- B. Diminuer la taille du jeu d'entraînement.
- C. Augmenter la taille du jeu d'entraînement.
- D. Enlever les données de validation.

**✓ Réponse : C**

---

### Question 4
Vous avez deux projets de machine learning : l'un consiste à regrouper automatiquement des clients selon leur comportement d'achat, l'autre à prédire si un email est un spam ou non. Quelle affirmation décrit correctement la différence entre les deux approches ?
- A. Les deux projets utilisent des données étiquetées et relèvent de l'apprentissage supervisé.
- B. Le regroupement automatique se fait sans étiquettes, alors que la détection de spam utilise des étiquettes connues.
- C. La classification ne nécessite jamais de données étiquetées, contrairement au clustering.
- D. Les deux projets sont des exemples d'apprentissage non supervisé.

**✓ Réponse : B**

---

### Question 5
Parmi les trois courbes ci-dessous représentant l'ajustement d'un modèle sur des données d'entraînement, laquelle illustre un cas d'underfitting ?
- A. Figure 1 (modèle simple, perte élevée)
- B. Figure 2 (modèle équilibré, bonne perte)
- C. Figure 3 (modèle complexe, surapprentissage)
- D. Aucune

**✓ Réponse : A**

**Explication** : L'underfitting se produit avec un modèle trop simple qui n'apprend pas les patterns.

---

### Question 6
Quel type d'apprentissage correspond à un robot qui apprend à maximiser ses récompenses après chaque action réussie ?
- A. Apprentissage supervisé
- B. Apprentissage non supervisé
- C. Apprentissage par renforcement
- D. Apprentissage semi-supervisé

**✓ Réponse : C**

---

### Question 7
Dans la méthodologie CRISP-DM, quelle phase suit immédiatement la phase "Business Understanding" ?
- A. Modeling
- B. Data Understanding
- C. Data Preparation
- D. Evaluation

**✓ Réponse : B**

---

### Question 8
Quel est le principal objectif de la phase "Data Preparation" en CRISP-DM ?
- A. Visualiser les données brutes
- B. Transformer données brutes en format exploitable pour la modélisation
- C. Évaluer la performance du modèle
- D. Déployer le modèle en production

**✓ Réponse : B**

---

### Question 9
En CRISP-DM, pourquoi peut-on revenir à la phase "Data Preparation" après la phase "Evaluation" ?
- A. C'est une erreur de méthodologie
- B. Parce que CRISP-DM est itératif et les résultats peuvent révéler des problèmes de préparation
- C. Parce qu'il faut toujours nettoyer les données deux fois
- D. Pour économiser du temps

**✓ Réponse : B**

---

### Question 10
Quelle est l'importance relative du preprocessing dans un projet ML selon CRISP-DM ?
- A. 10-20% du temps total
- B. 30-40% du temps total
- C. 60-80% du temps total
- D. Plus de 90% du temps total

**✓ Réponse : C**

**Explication** : La préparation des données est la phase la plus longue et critique.

---

## Questions 11-20 : Data Preprocessing

### Question 11
Si un dataset contient 50% de valeurs manquantes dans une colonne numérique fortement corrélée (>0.9) à une autre, la meilleure approche serait :
- A. Supprimer la colonne
- B. Imputer par la moyenne
- C. Imputer par la médiane
- D. Imputer par zéro

**✓ Réponse : A**

**Explication** : Avec 50% de valeurs manquantes, supprimer la colonne est plus sûr que d'inventer des données.

---

### Question 12
Quel encodeur produit une dimensionnalité proportionnelle au nombre de catégories distinctes ?
- A. Label Encoder
- B. One-Hot Encoder
- C. Ordinal Encoder
- D. Frequency Encoder

**✓ Réponse : B**

**Explication** : One-Hot crée une colonne binaire par catégorie, donc K catégories → K colonnes.

---

### Question 13
Lorsqu'on applique une PCA sans standardisation ou normalisation, quel effet risque-t-on d'observer ?
- A. Les composantes principales deviennent corrélées
- B. Les composantes principales changent de signe
- C. Les variables à grande échelle dominent la variance
- D. Le PCA échoue numériquement

**✓ Réponse : C**

**Explication** : Sans standardisation, les variables avec grande variance dominant le calcul de PCA.

---

### Question 14
Si l'on sait que certaines valeurs négatives dans la variable "âge" proviennent d'une erreur de saisie, quelle approche serait la plus adaptée pour gérer ces valeurs ?
- A. Supprimer toutes les lignes
- B. Imputer les âges négatifs par la médiane
- C. Encoder les âges négatifs comme catégorie spéciale
- D. Inverser le signe des âges négatifs

**✓ Réponse : B**

**Explication** : Supprimer toutes les lignes est excessif. Imputer par la médiane est approprié.

---

### Question 15
Quelle méthode est la plus efficace pour détecter les outliers dans un espace unidimensionnel ?
- A. Bar plot
- B. Z-score
- C. Analyse de densité
- D. Boxplot

**✓ Réponse : D** (ou B avec des nuances)

**Explication** : Boxplot est visuel et efficace. Z-score est mathématiquement rigoureux mais suppose normalité.

---

### Question 16
Pourquoi faut-il encoder avant de scaler ?
- A. Car les encodeurs dépendent de la moyenne
- B. Car le scaling ne supporte pas les strings
- C. Car le scaling dégrade les catégories
- D. Les deux B et C

**✓ Réponse : D**

**Explication** : Le scaling travaille sur nombres, donc il faut d'abord encoder les catégories.

---

### Question 17
Quelle méthode de feature engineering est adaptée pour extraire de l'information temporelle à partir d'une colonne date ?
- A. Extraction du jour de la semaine
- B. Extraction de jour, mois, année
- C. Extraction de la durée écoulée de cette date
- D. Extraction de la saisonnalité

**✓ Réponse : D** (ou B/C comme réponses partielles correctes)

**Explication** : Toutes les options sont valides. D est la plus générale.

---

### Question 18
Vous modélisez la détection de fraudes ; "montant transaction" contient quelques valeurs extrêmes (ex. 10 000 000) qui correspondent à des fraudes réelles et importantes. Quelle action est la moins appropriée si ces outliers sont des observations valides du phénomène que vous voulez détecter ?
- A. Supprimer ces lignes
- B. Utiliser des modèles robustes ou méthodes qui tolèrent les valeurs extrêmes
- C. Appliquer une transformation (ex. log) pour réduire l'effet des grands montants
- D. Imputer après vérification du sens métier et de l'objectif

**✓ Réponse : A**

**Explication** : Si ces outliers sont des fraudes vraies, les supprimer les ferait disparaître de nos données d'entraînement.

---

### Question 19
Quelle méthode d'encodage est la plus adaptée pour une variable catégorielle ordinale comme Faible < Moyen < Élevé ?
- A. One-Hot Encoding
- B. Label Encoding
- C. Encodage fréquentiel
- D. Binary Encoding

**✓ Réponse : B**

**Explication** : Label Encoding préserve l'ordre naturel : Faible=0, Moyen=1, Élevé=2.

---

### Question 20
Parmi ces stratégies d'imputation, laquelle est la plus robuste aux outliers ?
- A. Imputation par la moyenne
- B. Imputation par la médiane
- C. Imputation par le mode
- D. Imputation KNN

**✓ Réponse : B**

**Explication** : La médiane est robuste aux outliers, contrairement à la moyenne.

---

## Questions 21-30 : Data Preprocessing (Suite)

### Question 21
Quel type de normalisation met les données dans l'intervalle [0,1] ?
- A. Standardisation (Z-score)
- B. Min-Max Scaling
- C. Robust Scaling
- D. Log Transform

**✓ Réponse : B**

**Explication** : Min-Max Scaling : $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$

---

### Question 22
La standardisation Z-score est préférable au Min-Max Scaling lorsque :
- A. Les données contiennent des outliers
- B. Les données suivent une distribution normale
- C. Les données sont bornées entre deux valeurs
- D. Les données sont hautement asymétriques

**✓ Réponse : A**

**Explication** : Z-score est plus robuste aux outliers que Min-Max.

---

### Question 23
Sélection de caractéristiques : Quelle méthode de sélection de caractéristiques utilise directement les performances du modèle pour évaluer l'importance des variables ?
- A. Méthode basée sur les filtres
- B. Méthode basée sur les wrappers
- C. Méthode intégrée
- D. Méthode par imputation

**✓ Réponse : B**

**Explication** : Wrapper methods évaluent les features selon la performance du modèle.

---

### Question 24
Quelle méthode de sélection de features est la plus rapide ?
- A. Forward Selection
- B. Backward Elimination  
- C. Filter Methods
- D. Recursive Feature Elimination

**✓ Réponse : C**

**Explication** : Filter Methods ne nécessitent pas d'entraîner le modèle.

---

### Question 25
La régularisation L1 (Lasso) effectue automatiquement une forme de :
- A. Normalisation
- B. Sélection de features
- C. Imputation
- D. Encodage

**✓ Réponse : B**

**Explication** : Lasso pousse certains coefficients à zéro, effectuant une sélection de features.

---

### Question 26
Quel est le concept statistique derrière les valeurs manquantes MCAR ?
- A. Missing Completely At Random = manque sans lien avec les données
- B. Missing At Random = manque lié à d'autres variables
- C. Missing Not At Random = manque lié à sa propre valeur
- D. Aucun rapport

**✓ Réponse : A**

---

### Question 27
Si vous avez des données avec valeurs manquantes MAR (Missing At Random), quelle approche est déconseillée ?
- A. Suppressionlistwise (listwise deletion)
- B. Imputation simple
- C. MICE (Multiple Imputation)
- D. KNN Imputation

**✓ Réponse : A**

**Explication** : Listwise deletion introduit du biais pour données MAR.

---

### Question 28
Quel problème se pose quand on applique One-Hot Encoding sur une variable avec 1000 catégories distinctes ?
- A. Pas de problème majeur
- B. Curse of Dimensionality (trop de colonnes)
- C. Les catégories deviennent non ordonnées
- D. Les valeurs manquantes ne peuvent pas être imputées

**✓ Réponse : B**

---

### Question 29
Pour une variable catégorique non ordinale (ex: couleur = {Rouge, Bleu, Vert}), pourquoi One-Hot Encoding est préférable à Label Encoding ?
- A. C'est plus rapide
- B. Parce qu'une valeur numérique sugère un ordre artificiel
- C. Parce que ça réduit les dimensions
- D. Parce que ça améliore la vitesse du modèle

**✓ Réponse : B**

---

### Question 30
Quel est l'ordre recommandé pour le preprocessing ?
- A. Encoding → Normalisation → Sélection de features
- B. Normalisation → Encoding → Sélection de features
- C. Encoding → Sélection de features → Normalisation
- D. It depends on the algorithm

**✓ Réponse : A** ou **C**

**Explication** : Généralement : Nettoyage → Encoding → Normalisation → Feature Selection

---

## Questions 31-40 : ACP/PCA

### Question 31
Dans une ACP sur les notes d'étudiants, la première composante explique 65% de la variance. Que cela signifie ?
- A. La composante résume la majorité des différences entre étudiants
- B. Les matières sont indépendantes
- C. L'ACP n'est pas valide
- D. Les données sont bruitées

**✓ Réponse : A**

---

### Question 32
Si deux variables sont très proches dans le cercle des corrélations :
- A. Elles sont corrélées négativement
- B. Elles sont indépendantes
- C. Elles sont fortement corrélées positivement
- D. Elles doivent être supprimées

**✓ Réponse : C**

---

### Question 33
Dans un scree plot, la chute des valeurs propres après la 2e composante indique :
- A. Garder toutes les composantes
- B. Les deux premières suffisent
- C. L'ACP est invalide
- D. Supprimer la première composante

**✓ Réponse : B**

---

### Question 34
Sur un plan factoriel, les individus proches les uns des autres :
- A. Ont un profil similaire
- B. Appartiennent à la même classe
- C. Ont des valeurs extrêmes
- D. Sont des outliers

**✓ Réponse : A**

---

### Question 35
Deux variables, "âge" et "revenu", sont à 180° sur le cercle. Conclusion ?
- A. Corrélées positivement
- B. Indépendantes
- C. Corrélées négativement
- D. Identiques

**✓ Réponse : C**

---

### Question 36
Une entreprise agroalimentaire fait une ACP sur les caractéristiques chimiques. Quel sera son objectif ?
- A. Prédire la vente future
- B. Identifier des groupes de produits proches
- C. Supprimer les variables
- D. Remplacer les mesures par du texte

**✓ Réponse : B**

---

### Question 37
Une entreprise de transport applique une ACP sur temps de trajet, vitesse et nombre d'arrêts. PC1 oppose "vitesse" et "nombre d'arrêts". Que représente PC1 ?
- A. Les trajets courts
- B. L'efficacité du trajet
- C. Les valeurs extrêmes
- D. Les trajets aléatoires

**✓ Réponse : B**

**Explication** : PC1 capture la variation la plus importante = l'efficacité du trajet.

---

### Question 38
Une ACP avant clustering permet :
- A. Réduire le bruit et améliorer la séparation
- B. Augmenter la dimension
- C. Supprimer les similarités
- D. Désorganiser les clusters

**✓ Réponse : A**

---

### Question 39
PC1 explique 55% et PC2 25% de la variance. Conclusion ?
- A. Ces deux composantes représentent 0.8 de l'information initiale
- B. Supprimer PC2
- C. Variables non corrélées
- D. Ces deux composantes représentent 0.2 de l'information initiale

**✓ Réponse : A**

**Explication** : 55% + 25% = 80% = 0.8 de l'information.

---

### Question 40
Une variable bien représentée sur PC1 mais pas sur PC2 signifie :
- A. Contribue surtout à PC1
- B. Contribue surtout à PC2
- C. La variable est inutile
- D. Contribue à PC1 et PC2 également

**✓ Réponse : A**

---

## Questions 41-50 : ACP/PCA (Suite)

### Question 41
La standardisation des données est-elle nécessaire avant une ACP ?
- A. Non, elle complique le calcul
- B. Oui, obligatoire, sinon les variables à grande échelle dominent
- C. Oui, mais seulement pour variables catégorielles
- D. Non, l'ACP la fait automatiquement

**✓ Réponse : B**

---

### Question 42
Quelle est la propriété mathématique fondamentale des axes principaux en ACP ?
- A. Ils sont parallèles
- B. Ils sont orthogonaux (perpendiculaires) entre eux
- C. Ils sont alignés avec les axes de coordonnées originaux
- D. Ils maximisent la covariance

**✓ Réponse : B**

---

### Question 43
En ACP, pourquoi les composantes principales sont-elles décorrélées ?
- A. Par construction : ce sont des vecteurs propres orthogonaux de la matrice de covariance
- B. C'est un processus aléatoire
- C. Parce qu'on les standardise
- D. Parce qu'on les normalise

**✓ Réponse : A**

---

### Question 44
Quel indicateur du scree plot indique le nombre de composantes à conserver ?
- A. Le point le plus haut
- B. Le "coude" : où la pente change significativement
- C. Le point le plus bas
- D. Le dernier point

**✓ Réponse : B**

---

### Question 45
Si la variance cumulée atteint 95% avec 10 composantes sur 50 initiales, peut-on réduire à 10 ?
- A. Non, on perd 5% d'information
- B. Oui, 95% est généralement acceptable (96-99% est excellent)
- C. Non, il faut garder toutes les 50
- D. Oui, mais seulement si les données sont très bruitées

**✓ Réponse : B**

---

### Question 46
Qu'indique une variable très proche du centre du cercle de corrélations ?
- A. La variable est très importante
- B. La variable est corrélée à toutes les autres
- C. La variable n'est pas bien représentée sur PC1-PC2 (variance sur autres PC)
- D. Il y a une erreur dans l'ACP

**✓ Réponse : C**

---

### Question 47
Comment interpréter un individu situé dans la direction d'une variable sur le biplot ?
- A. L'individu appartient à cette catégorie
- B. L'individu a une valeur élevée pour cette variable
- C. L'individu ne contribue pas à cette variable
- D. L'individu est un outlier

**✓ Réponse : B**

---

### Question 48
L'ACP peut-elle être utilisée pour compresser des images numériques ?
- A. Non, elle ne fonctionne que pour données tabulaires
- B. Oui, car elle réduit les dimensions (pixels) tout en conservant l'info visuellement importante
- C. Oui, mais c'est plus lent que JPEG
- D. Non, les images nécessitent des techniques spécialisées

**✓ Réponse : B**

---

### Question 49
Si un algorithme de classification prend trop de temps avec 200 features, quel serait le bénéfice d'appliquer PCA ?
- A. Aucun bénéfice, la classification sera plus lente
- B. Réduire les dimensions (ex: 200 → 20 PC), accélérer entraînement
- C. Augmenter la précision
- D. Éliminer tous les outliers

**✓ Réponse : B**

---

### Question 50
La PCA est-elle un algorithme supervisé ou non supervisé ?
- A. Supervisé : elle utilise les étiquettes (y)
- B. Non supervisé : elle n'utilise que les features (X)
- C. Semi-supervisé
- D. Non applicable

**✓ Réponse : B**

---

## Questions 51-60 : K-Means

### Question 51
Quel est l'impact des outliers sur K-means ?
- A. Ils améliorent la cohérence des clusters.
- B. Ils déplacent les centroïdes vers les régions peu denses.
- C. Ils n'affectent pas les résultats.
- D. Ils réduisent le nombre d'itérations nécessaires.

**✓ Réponse : B**

---

### Question 52
Quelle condition garantit la convergence de l'algorithme K-Means ?
- A. Quand la fonction de coût commence à osciller
- B. Quand le nombre d'itérations atteint 100
- C. Quand les centroïdes cessent de changer ou que les attributions ne changent plus
- D. Quand la variance intra-cluster augmente

**✓ Réponse : C**

---

### Question 53
Quelle est la principale limitation de la méthode du coude (elbow method) ?
- A. Le "coude" n'est pas toujours clairement identifiable visuellement
- B. Elle nécessite une normalisation préalable
- C. Elle ne fonctionne que pour K < 10
- D. Elle est incompatible avec K-means++

**✓ Réponse : A**

---

### Question 54
Quelle hypothèse implicite K-means fait-il sur la forme des clusters ?
- A. Les clusters sont sphériques avec des variances approximativement égales
- B. Les clusters ont des formes elliptiques avec des axes alignés
- C. Les clusters peuvent avoir n'importe quelle forme convexe
- D. Les clusters suivent une distribution gaussienne multivariée

**✓ Réponse : A**

---

### Question 55
Lorsqu'on applique K-means à un jeu de données non normalisées où les attributs ont des échelles différentes, quel est le risque principal ?
- A. L'algorithme ne convergera pas.
- B. Les centroïdes seront initialisés sur les mêmes points.
- C. Le coefficient de silhouette sera systématiquement égal à 1.
- D. L'attribut avec la plus grande variance dominera la formation des clusters.

**✓ Réponse : D**

---

### Question 56
Observez le graphique ci-dessous montrant l'évolution de l'inertie en fonction de K. Quel est le nombre optimal de clusters selon la méthode du coude ?
- A. K = 2
- B. K = 3
- C. K = 4
- D. K = 5

**✓ Réponse : B** ou **C** (dépend du graphique)

---

### Question 57
Sur le diagramme suivant, trois points sont représentés dont deux centroïdes C₁ et C₂. En utilisant la distance euclidienne, à quel cluster le point P sera-t-il assigné lors de la prochaine itération ?
Coordonnées : C₁ (1,1), C₂ (3,4), P (5,3)
- A. Cluster 1 (C₁)
- B. Cluster 2 (C₂)
- C. Aucun (équidistant)
- D. Impossible à déterminer

**✓ Réponse : B**

**Calcul** : 
- Distance P-C₁ : √[(5-1)² + (3-1)²] = √[16+4] = √20 ≈ 4.47
- Distance P-C₂ : √[(5-3)² + (3-4)²] = √[4+1] = √5 ≈ 2.24
- P est plus proche de C₂

---

### Question 58
Quel est le problème principal de la méthode d'initialisation aléatoire des centroïdes en K-means ?
- A. Elle donne toujours des mauvais résultats
- B. Elle converge trop lentement
- C. Elle peut converger vers un minimum local moins bon
- D. Elle ne fonctionne que pour K petit

**✓ Réponse : C**

---

### Question 59
K-means++ améliore l'initialisation en :
- A. Choisissant les centroïdes aléatoirement
- B. Choisissant les centroïdes espacés pour éviter initialisation mauvaise
- C. Utilisant le centre du dataset comme premier centroïde
- D. Utilisant une grille régulière de centroïdes

**✓ Réponse : B**

---

### Question 60
La silhouette est une métrique qui mesure :
- A. Combien de clusters il faut
- B. La proximité des points à leur centroïde
- C. La cohésion intra-cluster ET la séparation inter-cluster
- D. La vitesse de convergence

**✓ Réponse : C**

---

## Questions 61-70 : DBSCAN

### Question 61
Lorsqu'on augmente la valeur de ε tout en gardant MinPts constant, quel effet est le plus probable sur le résultat du clustering ?
- A. Le nombre de clusters détectés diminue
- B. Le nombre de points considérés comme bruit augmente
- C. Les clusters deviennent plus petits et plus nombreux
- D. Les frontières entre clusters deviennent plus nettes

**✓ Réponse : A**

**Explication** : Augmenter ε fait fusionner des clusters, donc le nombre diminue.

---

### Question 62
Si MinPts est choisi trop grand par rapport à la densité des données, alors :
- A. L'algorithme formera de nombreux petits clusters
- B. La plupart des points seront considérés comme du bruit
- C. Le temps de calcul diminuera
- D. Les clusters seront plus compacts mais plus nombreux

**✓ Réponse : B**

---

### Question 63
Parmi les propositions suivantes, laquelle n'est pas vraie à propos des points frontières dans DBSCAN ?
- A. Ils appartiennent à un cluster
- B. Ils sont voisins d'au moins un point central
- C. Ils peuvent devenir des points centraux si MinPts diminue
- D. Ils peuvent initier la formation d'un nouveau cluster

**✓ Réponse : D**

**Explication** : Seuls les points centraux initient les clusters DBSCAN.

---

### Question 64
Quel est le principal inconvénient de l'utilisation du k-distance plot pour choisir ε ?
- A. Il dépend du nombre de points et du bruit dans les données
- B. Il donne toujours la même valeur de ε
- C. Il nécessite de connaître le nombre de clusters
- D. Il ne fonctionne que pour les données catégorielles

**✓ Réponse : A**

---

### Question 65
Dans une application réelle, DBSCAN renvoie un grand nombre de points considérés comme du bruit. Quelle action est la plus appropriée ?
- A. Réduire ε
- B. Augmenter ε
- C. Réduire MinPts
- D. Utiliser K-Means à la place

**✓ Réponse : C** (ou **B** avec nuances)

**Explication** : Réduire MinPts rendra plus facile d'être un point central, réduisant le bruit.

---

### Question 66
Pour un jeu de données avec D = 4 dimensions, quelle valeur minimale de MinPts est recommandée selon la règle pratique ?
- A. 2
- B. 4
- C. 5
- D. 8

**✓ Réponse : C**

**Explication** : Règle pratique : MinPts ≥ D + 1, donc pour D=4 → MinPts ≥ 5.

---

### Question 67
DBSCAN est plus efficace que les algorithmes hiérarchiques lorsque :
- A. Les données contiennent des formes irrégulières
- B. Le nombre de clusters est connu
- C. Les données sont déjà normalisées
- D. Tous les clusters ont la même taille

**✓ Réponse : A**

---

### Question 68
Quelle propriété distingue principalement DBSCAN de K-means ?
- A. DBSCAN est plus rapide
- B. K-means ne suppose pas sphéricité, DBSCAN la suppose
- C. DBSCAN peut trouver des clusters de formes arbitraires, K-means suppose sphériques
- D. DBSCAN nécessite étiquetage des données

**✓ Réponse : C**

---

### Question 69
Un "point central" dans DBSCAN est un point qui :
- A. Est au centre du dataset
- B. A au moins MinPts voisins dans un rayon ε
- C. Est à proximité du centroïde
- D. Belong to the convex hull

**✓ Réponse : B**

---

### Question 70
Pourquoi DBSCAN génère souvent du bruit (points non assignés à un cluster) ?
- A. C'est une limitation de l'algorithme
- B. Parce que certains points n'ont pas suffisamment de voisins (< MinPts) dans rayon ε
- C. Parce qu'on utilise toujours mal les paramètres ε et MinPts
- D. Parce que les données sont mal préparées

**✓ Réponse : B**

---

## Questions 71-80 : KNN & SVM

### Question 71
Quel est le principal avantage de KNN pour la classification ?
- A. C'est un algorithme très rapide à entraîner
- B. Il capture des relations non linéaires complexes
- C. Il demande peu de preprocessing
- D. Il fonctionne très bien en haute dimension

**✓ Réponse : B**

---

### Question 72
Quel est le principal inconvénient de KNN ?
- A. Il capture mal les relations non linéaires
- B. Il est sensible al curse of dimensionality
- C. Il demande trop d'entraînement
- D. Il ne fonctionne qu'avec données catégorielles

**✓ Réponse : B**

---

### Question 73
Pourquoi appelle-t-on KNN un algorithme "lazy" ?
- A. Il ne converge pas rapidement
- B. Il ne perf pas d'apprentissage (entraînement) : tout se passe à la prédiction
- C. Il ignore les données bruitées
- D. Il est très lent

**✓ Réponse : B**

---

### Question 74
Comment choisir la valeur de K dans KNN ?
- A. Toujours K=3
- B. Toujours K=5
- C. Par cross-validation en testant différentes valeurs
- D. K = racine carrée du nombre d'observations

**✓ Réponse : C**

---

### Question 75
Quel est le principal objectif de SVM (Support Vector Machine) ?
- A. Minimiser l'erreur d'entraînement uniquement
- B. Maximiser la marge entre classes tout en minimisant erreur
- C. Créer une hyperplan qui passe par tous les points
- D. Réduire le nombre de features

**✓ Réponse : B**

---

### Question 76
Quel est le rôle du paramètre C dans SVM ?
- A. Contrôle la variance de la marge
- B. Contrôle le compromis entre marge maximale et erreurs d'entraînement
- C. Contrôle le type de kernel
- D. Contrôle le nombre de support vectors

**✓ Réponse : B**

---

### Question 77
Quel type de kernel SVM utiliser si les données ne sont pas linéairement séparables ?
- A. Linear kernel
- B. Polynomial ou RBF kernel
- C. Sigmoid kernel
- D. No kernel possible

**✓ Réponse : B**

---

### Question 78
Quelle est la relation entre le nombre de support vectors et la complexité du modèle SVM ?
- A. Plus de support vectors = modèle plus simple
- B. Plus de support vectors = modèle plus complexe
- C. Pas de relation
- D. Dépend du type de kernel

**✓ Réponse : B**

---

### Question 79
Pour quoi la normalisation est-elle particulièrement importante en SVM ?
- A. Elle n'est pas importante
- B. Parce que SVM est basé sur distances, et distances peuvent être faussées par échelles différentes
- C. Parce qu'elle accélère la convergence
- D. Parce qu'elle réduit le nombre de support vectors

**✓ Réponse : B**

---

### Question 80
Quel type de problème SVM gère naturellement mieux que KNN ?
- A. Données muy bruitées
- B. Données en haute dimension
- C. Données non linéairement séparables (avec kernel)
- D. Données avec classes très déséquilibrées

**✓ Réponse : C**

---

## Questions 81-90 : Concepts Transversaux

### Question 81
Qu'est-ce que la validation croisée (cross-validation) ?
- A. Une méthode pour évaluer la stabilité d'un modèle en le testant sur différents sous-ensembles
- B. Une méthode pour augmenter la taille des données
- C. Une technique d'encodage
- D. Un type de normalisation

**✓ Réponse : A**

---

### Question 82
Quelle métrique est utilisée pour évaluer la qualité des clusters en clustering ?
- A. Accuracy
- B. Precision et Recall
- C. Silhouette Score
- D. F1-Score

**✓ Réponse : C** (ou Inertie, Davies-Bouldin Index)

---

### Question 83
Qu'est-ce que l'overfitting ?
- A. Quand le modèle n'apprend pas bien les données d'entraînement
- B. Quand le modèle mémorise les données d'entraînement et généralise mal
- C. Quand il y a trop de claases
- D. Quand on utilise trop de features

**✓ Réponse : B**

---

### Question 84
Quel est le rôle de la matrice de confusion en classification ?
- A. Confondre les classes
- B. Montrer visuellement les vrais positifs, vrais négatifs, faux positifs, faux négatifs
- C. Normaliser les données
- D. Réduire les dimensions

**✓ Réponse : B**

---

### Question 85
Qu'est-ce que le "No Free Lunch Theorem" en ML ?
- A. Aucun théorème n'existe en ML
- B. Il n'y a pas d'algorithme parfait pour tous les problèmes
- C. Un algorithme gratuit pour résoudre tous les problèmes
- D. Les données doivent toujours être gratuites

**✓ Réponse : B**

---

### Question 86
Comment gérer un problème de déséquilibre de classes (class imbalance) ?
- A. Ignorer les classes minoritaires
- B. Augmenter les données de classe minoritaire (oversampling) ou réduire majorité (undersampling)
- C. Toujours utiliser accuracy comme métrique
- D. Il n'existe pas de solution

**✓ Réponse : B**

---

### Question 87
Qu'est-ce que le bias-variance tradeoff ?
- A. Un compromis entre underfitting et overfitting
- B. Une technique de normalisation
- C. Un type de données manquantes
- D. Une métrique d'évaluation

**✓ Réponse : A**

---

### Question 88
Quel est l'objectif principal de la validation sur un test set ?
- A. Entraîner le modèle
- B. Évaluer la performance du modèle sur données non vues pendant l'entraînement
- C. Valider la normalisation des données
- D. Choisir les hyperparamètres

**✓ Réponse : B**

---

### Question 89
Qu'est-ce que la feature importance dans un modèle arborescent (comme Random Forest) ?
- A. La taille de la feature
- B. La contribution d'une feature à la réduction de l'impureté
- C. Le nombre de features utilisées
- D. Le type de feature

**✓ Réponse : B**

---

### Question 90
Quel est le risque principal du data leakage ?
- A. Les données prennent trop de place disque
- B. Des informations du test set filtrent dans l'entraînement, créant faux résultats
- C. Les données deviennent illégales
- D. Les features sont mal encodées

**✓ Réponse : B**

---

## Questions 91-100 : Cas Pratiques

### Question 91
Vous construisez un modèle de prédiction de churn client. Les données contiennent 10 000 clients, dont 500 partis (5%). Quel problème allez-vous rencontrer ?
- A. Données insuffisantes
- B. Classes très déséquilibrées (class imbalance)
- C. Dimensions trop hautes
- D. Données corrompues

**✓ Réponse : B**

---

### Question 92
Pour ce modèle de churn, quelle métrique est la plus appropriée ?
- A. Accuracy (même un modèle qui dit jamais de churn aurait 95% accuracy)
- B. Precision et Recall
- C. MAE (Mean Absolute Error)
- D. Correlation

**✓ Réponse : B**

---

### Question 93
Vous avez un dataset avec 1000 samples et 500 features. Quel problème principal risquez-vous de rencontrer ?
- A. Trop de données
- B. Curse of dimensionality (overfit probable)
- C. Les features sont corrélées
- D. Les données sont bruitées

**✓ Réponse : B**

---

### Question 94
Pour résoudre ce problème de haute dimensionnalité, que feriez-vous en premier ?
- A. Appliquer immédiatement K-means
- B. Appliquer une réduction dimensionnalité (PCA ou feature selection)
- C. Augmenter la taille du dataset
- D. Utiliser un modèle plus simple

**✓ Réponse : B**

---

### Question 95
En analysant les comportements d'achat de clients (sans étiquettes), quel algorithme choisiriez-vous ?
- A. KNN
- B. SVM
- C. K-Means ou DBSCAN
- D. Régression logistique

**✓ Réponse : C**

---

### Question 96
Après clustering, vous constatez que DBSCAN génère beaucoup de bruit (points non assignés). Quelle action prioritaire ?
- A. Utiliser K-means à la place
- B. Réduire MinPts progressivement pour diminuer le bruit
- C. Augmenter ε pour fusionner les petits clusters
- D. Supprimer les points de bruit

**✓ Réponse : C** (ou **B**)

---

### Question 97
Vous devez classifier 100 images de fruits (pommes, oranges, bananes) avec SVM. Quel est le pretraitement essentiel ?
- A. One-Hot Encoder les images
- B. Redimensionner les images à taille uniforme ET normaliser les pixels [0,1]
- C. Augmenter le dataset
- D. Rien, SVM fonctionne sur les images brutes

**✓ Réponse : B**

---

### Question 98
Pour prédire si un email est spam ou non, quel type d'apprentissage est approprié ?
- A. Clustering (non supervisé)
- B. Classification supervisée (KNN, SVM, etc.)
- C. Réduction dimensionnalité
- D. Renforcement

**✓ Réponse : B**

---

### Question 99
Vous trouvez que votre modèle KNN a un accuracy de 99% sur données d'entraînement mais 70% sur test. Diagnostique ?
- A. Le modèle est excellent
- B. Overfitting : K trop petit, modèle trop complexe
- C. Underfitting : K trop grand
- D. Les données test sont mauvaises

**✓ Réponse : B**

---

### Question 100
Quelle est l'étape la plus critique dans le pipeline ML (CRISP-DM) qui détermine 80% de la qualité finale ?
- A. La modélisation (choix algorithme)
- B. L'évaluation (qu mesure-t-on ?)
- C. La préparation des données (cleaning, encoding, normalisation)
- D. Le déploiement

**✓ Réponse : C**

**Explication** : "Garbage in, garbage out" - la qualité des données préparées détermine la qualité du modèle.

---

## 📊 Récapitulatif par Thèmes

| Thème | Questions | Nombre |
|-------|-----------|--------|
| CRISP-DM | 1-10 | 10 |
| Data Preprocessing | 11-30 | 20 |
| ACP/PCA | 31-50 | 20 |
| K-Means | 51-60 | 10 |
| DBSCAN | 61-70 | 10 |
| KNN & SVM | 71-80 | 10 |
| Concepts Transversaux | 81-90 | 10 |
| Cas Pratiques | 91-100 | 10 |
| **TOTAL** | | **100** |

---

## 🎯 Stratégie de Révision

1. **Jour 1** : QCU 1-30 (CRISP-DM + Data Preprocessing)
2. **Jour 2** : QCU 31-60 (ACP + K-Means)
3. **Jour 3** : QCU 61-90 (DBSCAN + KNN/SVM + Concepts)
4. **Jour 4** : QCU 91-100 + Révision des erreurs

---

**Bonne chance à l'examen ! 🍀**
