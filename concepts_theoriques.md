# 📖 Concepts Théoriques - Machine Learning

**Guide théorique complet pour l'examen**

---

## Table des Matières Théorique

1. [CRISP-DM - Méthodologie](#1-crisp-dm---méthodologie)
2. [Data Preprocessing - Prétraitement](#2-data-preprocessing---prétraitement)
3. [ACP (PCA) - Réduction Dimensionnalité](#3-acp-pca---réduction-dimensionnalité)
4. [K-Nearest Neighbors (KNN)](#4-k-nearest-neighbors-knn)
5. [Support Vector Machine (SVM)](#5-support-vector-machine-svm)
6. [K-means Clustering](#6-k-means-clustering)
7. [DBSCAN Clustering](#7-dbscan-clustering)
8. [Concepts Transversaux](#8-concepts-transversaux)

---

## 1. CRISP-DM - Méthodologie

### 🎯 Définition

**CRISP-DM** = Cross-Industry Standard Process for Data Mining

Méthodologie standard et structurée pour mener à bien des projets de Data Mining et Machine Learning, développée en 1996 par un consortium d'entreprises.

### 📐 Caractéristiques Fondamentales

#### 1. **Processus Itératif et Non Linéaire**

**Concept clé** : CRISP-DM n'est PAS une cascade (waterfall)

```
┌─────────────────────────────────────────┐
│  Business Understanding                 │
│         ↓         ↑                     │
│  Data Understanding  ← Retours possibles│
│         ↓         ↑                     │
│  Data Preparation    ← Retours possibles│
│         ↓         ↑                     │
│    Modeling          ← Retours possibles│
│         ↓         ↑                     │
│   Evaluation                            │
│         ↓                               │
│   Deployment                            │
└─────────────────────────────────────────┘
```

**Pourquoi itératif ?**
- Les résultats d'une phase peuvent nécessiter de revoir une phase précédente
- L'évaluation peut révéler des problèmes de préparation
- La modélisation peut nécessiter plus de features (retour à Data Preparation)

#### 2. **Les 6 Phases en Détail**

##### Phase 1 : Business Understanding

**Objectif théorique** : Traduire un problème métier en problème de ML

**Questions fondamentales :**
1. Quel est l'objectif métier ?
2. Quels sont les critères de succès ?
3. Quelles sont les contraintes (temps, budget, légales) ?
4. Quels risques ?

**Output** : 
- Document de spécifications
- Définition des objectifs ML
- Critères d'évaluation du succès

**Exemple théorique** :
```
Problème métier : "Réduire le taux de désabonnement clients"
        ↓
Problème ML : "Prédire quels clients vont se désabonner (classification binaire)"
        ↓
Critère succès : "Identifier 80% des futurs désabonnés avec précision >70%"
```

##### Phase 2 : Data Understanding

**Objectif théorique** : Comprendre la structure, qualité et potentiel des données

**Activités théoriques :**

1. **Collection des données**
   - Sources : bases, APIs, fichiers, scraping
   - Format : CSV, JSON, SQL, images...

2. **Description des données**
   - Nombre d'observations et features
   - Types de variables (numériques, catégorielles, texte, dates)
   - Ranges et distributions

3. **Exploration des données**
   - Statistiques descriptives
   - Visualisations (histogrammes, boxplots, scatter plots)
   - Recherche de patterns

4. **Vérification qualité**
   - Valeurs manquantes (pourcentage, pattern)
   - Outliers (aberrations)
   - Incohérences (âge négatif, dates futures)
   - Duplicatas

**Output** : Rapport de qualité des données

##### Phase 3 : Data Preparation

**Objectif théorique** : Transformer données brutes en dataset prêt pour ML

**Sous-tâches théoriques :**

1. **Sélection de données**
   - Quelles tables/colonnes utiliser ?
   - Filtrage des observations pertinentes

2. **Nettoyage**
   - Traitement valeurs manquantes (suppression, imputation)
   - Correction erreurs
   - Gestion duplicatas
   - Traitement outliers

3. **Construction de features** (Feature Engineering)
   - Création nouvelles variables dérivées
   - Transformations mathématiques
   - Agrégations
   - Extraction d'informations (dates → jour/mois/année)

4. **Intégration de données**
   - Jointures de tables
   - Fusion de sources multiples

5. **Formatage**
   - Encodage variables catégorielles
   - Normalisation/Standardisation
   - Reshaping (pivot, melt)

**Principe théorique fondamental** :
> "Garbage In, Garbage Out" - La qualité du modèle dépend de la qualité des données préparées

**Output** : Dataset final prêt pour modélisation

##### Phase 4 : Modeling

**Objectif théorique** : Sélectionner et appliquer techniques de modélisation

**Étapes théoriques :**

1. **Sélection de la technique**
   - Supervisé vs Non supervisé
   - Classification vs Régression vs Clustering
   - Choix algorithme selon problème

2. **Construction du plan de test**
   - Stratégie de validation (train/test, cross-validation)
   - Métriques d'évaluation

3. **Construction du modèle**
   - Entraînement sur données train
   - Tuning des hyperparamètres

4. **Évaluation du modèle**
   - Calcul des métriques
   - Analyse des erreurs

**Principe théorique** :
> "No Free Lunch Theorem" - Aucun algorithme n'est meilleur pour tous les problèmes

##### Phase 5 : Evaluation

**Objectif théorique** : Vérifier que le modèle atteint les objectifs métier

**Différence Modeling vs Evaluation** :
- **Modeling** : Évaluation technique (accuracy, MSE...)
- **Evaluation** : Évaluation métier (ROI, impact business, déploiement viable ?)

**Questions théoriques :**
1. Le modèle atteint-il les critères de succès définis ?
2. Y a-t-il des problèmes non détectés ?
3. Le modèle est-il robuste ?
4. Peut-on faire mieux ?

**Décisions possibles :**
- ✅ Déployer
- 🔄 Retour à une phase précédente
- ❌ Abandonner le projet

##### Phase 6 : Deployment

**Objectif théorique** : Intégrer le modèle dans l'environnement de production

**Formes de déploiement :**
1. **Batch predictions** : Prédictions périodiques (quotidien, hebdo)
2. **Real-time API** : Prédictions à la demande
3. **Edge deployment** : Modèle sur appareil (mobile, IoT)
4. **Report/Dashboard** : Visualisation des insights

**Considérations théoriques :**
- **Monitoring** : Surveiller performance en production
- **Maintenance** : Réentraîner périodiquement (data drift)
- **Documentation** : Guide utilisateur, documentation technique
- **Gouvernance** : Conformité, éthique, explicabilité

### 🔄 Nature Cyclique

**Concept théorique fondamental** :

CRISP-DM est un **cycle**, pas une ligne droite !

**Raisons des retours arrière :**
1. **Evaluation → Data Preparation** : Résultats insuffisants, besoin de plus de features
2. **Modeling → Data Understanding** : Découverte de patterns nécessitant plus d'exploration
3. **Deployment → Evaluation** : Problèmes en production détectés
4. **Any Phase → Business Understanding** : Changement des objectifs métier

**Exemple de cycle réel :**
```
Itération 1 : Business → Data → Prep → Model (KNN) → Eval (70% acc) → Retour Prep
Itération 2 : Prep (+ features) → Model (KNN) → Eval (75% acc) → Retour Model
Itération 3 : Model (SVM) → Eval (88% acc) → Retour Business (validation)
Itération 4 : Business (OK) → Deploy
```

### 📊 Pourquoi CRISP-DM ?

**Avantages théoriques :**
1. **Structuré** : Cadre clair, évite l'improvisation
2. **Flexible** : Adaptable à différents projets
3. **Itératif** : Amélioration continue
4. **Industry-standard** : Langage commun entre équipes
5. **Risk management** : Identifie problèmes tôt

**Alternatives existantes :**
- SEMMA (SAS)
- KDD (Knowledge Discovery in Databases)
- TDSP (Team Data Science Process - Microsoft)

---

## 2. Data Preprocessing - Prétraitement

### 🎯 Définition

**Data Preprocessing** = Ensemble de techniques pour transformer données brutes en format exploitable par algorithmes ML

### 📐 Principes Fondamentaux

#### Pourquoi le Preprocessing est Critique ?

**3 raisons théoriques :**

1. **Qualité des algorithmes ML**
   - Les algorithmes supposent données "propres"
   - Sensibles au bruit, valeurs manquantes, échelles

2. **Performance computationnelle**
   - Données non normalisées → convergence lente
   - Features inutiles → temps calcul gaspillé

3. **Interprétabilité**
   - Features encodées correctement → résultats compréhensibles

> **Principe** : "Data Science is 80% Preparation, 20% Modeling"

---

### A. Théorie des Valeurs Manquantes

#### Types de Valeurs Manquantes (Taxonomie de Rubin)

**1. MCAR (Missing Completely At Random)**
- Valeurs manquantes **sans lien** avec les données
- Exemple : Panne capteur aléatoire
- **Impact** : Perte de puissance statistique, mais pas de biais
- **Traitement** : Suppression acceptable

**2. MAR (Missing At Random)**
- Valeurs manquantes **liées à d'autres variables observées**
- Exemple : Hommes répondent moins à question sur salaire
- **Impact** : Biais si non traité
- **Traitement** : Imputation conditionnelle

**3. MNAR (Missing Not At Random)**
- Valeurs manquantes **liées à la valeur manquante elle-même**
- Exemple : Riches ne déclarent pas leur revenu
- **Impact** : Biais sévère
- **Traitement** : Difficile, nécessite modélisation explicite

#### Stratégies d'Imputation - Fondements Théoriques

**1. Suppression (Deletion)**

**Listwise deletion** : Supprimer toute ligne avec NA
- ✅ Simple
- ✅ Pas de biais si MCAR
- ❌ Perte d'information
- ❌ Réduit taille échantillon

**Pairwise deletion** : Utiliser données disponibles par paire
- ✅ Conserve plus de données
- ❌ Matrices de corrélation incohérentes

**Règle pratique** : Acceptable si < 5% données manquantes

**2. Imputation Simple**

**Par la moyenne** : $x_{missing} = \bar{x}$
- ✅ Simple, rapide
- ❌ Réduit variance
- ❌ Sensible aux outliers
- **Quand ?** Distribution normale, peu d'outliers

**Par la médiane** : $x_{missing} = \text{median}(x)$
- ✅ Robuste aux outliers
- ✅ Préserve mieux distribution
- **Quand ?** Distribution asymétrique, présence outliers

**Par le mode** : $x_{missing} = \text{mode}(x)$
- **Quand ?** Variables catégorielles

**Par constante** : $x_{missing} = 0$ ou valeur spécifique
- **Quand ?** 0 a du sens métier (ex: nb transactions)

**3. Imputation Avancée**

**KNN Imputation**
- Impute par moyenne des K voisins les plus proches
- Théorie : Points similaires ont valeurs similaires
- ✅ Capture relations entre features
- ❌ Coûteux computationnellement

**Imputation par régression**
- Prédire valeur manquante par régression sur autres variables
- ✅ Utilise corrélations
- ❌ Suppose linéarité

**MICE (Multiple Imputation by Chained Equations)**
- Imputation itérative de chaque variable
- Génère plusieurs datasets imputés
- ✅ Capture incertitude
- ❌ Complexe

---

### B. Théorie de l'Encodage

#### Fondements Mathématiques

**Problème** : Les algorithmes ML travaillent avec des nombres, pas des strings

**Solution** : Encoder catégories en nombres de manière appropriée

#### Types d'Encodage - Analyse Théorique

**1. Label Encoding**

**Principe** : Mapper chaque catégorie à un entier

$$\text{Catégorie} \rightarrow \mathbb{Z}$$

Exemple : {Rouge, Vert, Bleu} → {0, 1, 2}

**Problème théorique** :
- Crée **ordre artificiel** : Rouge < Vert < Bleu ?
- Modèles peuvent interpréter distances : distance(Rouge, Bleu) = 2

**Quand utiliser ?**
- ✅ Variable **ordinale** (ordre naturel existe)
- ✅ Variable **cible** (y) pour classification
- ❌ Variable **nominale** (pas d'ordre naturel)

**2. One-Hot Encoding**

**Principe** : Créer une colonne binaire par catégorie

$$\text{Catégorie} \rightarrow \{0,1\}^n$$

Exemple :
```
Couleur        →    Rouge  Vert  Bleu
Rouge               1      0     0
Vert                0      1     0
Bleu                0      0     1
```

**Propriétés mathématiques** :
- Vecteurs orthogonaux : $\vec{Rouge} \cdot \vec{Vert} = 0$
- Distance euclidienne entre toutes paires = $\sqrt{2}$
- Pas d'ordre artificiel

**Problème : Curse of Dimensionality**
- Si K catégories → K nouvelles colonnes
- Si K = 1000 (villes France) → 1000 colonnes !

**Dummy Trap** :
- K colonnes créent multicolinéarité : $\sum x_i = 1$
- Solution : Drop first ou drop_first=True (K-1 colonnes suffisent)

**3. Ordinal Encoding**

**Principe** : Label encoding avec **ordre spécifié**

$$\text{Catégorie ordonnée} \rightarrow \mathbb{Z} \text{ avec ordre préservé}$$

Exemple : {Faible=0, Moyen=1, Élevé=2}

**Justification théorique** :
- Préserve ordre naturel
- Distance a du sens : Élevé - Faible = 2

**4. Target Encoding**

**Principe** : Encoder par la moyenne de la cible pour cette catégorie

$$\text{Encode}(c) = \mathbb{E}[y | x = c]$$

Exemple :
```
Ville     → Moyenne salaire dans cette ville
Paris     → 45000
Lyon      → 38000
Marseille → 35000
```

**Avantage** :
- Une seule colonne (évite curse of dimensionality)
- Capture relation avec cible

**Problème : Data Leakage**
- Utilise information de y pour encoder X
- Risque d'overfitting
- **Solution** : Target encoding sur données OUT-OF-FOLD

---

### C. Théorie de la Normalisation

#### Fondements Mathématiques

**Problème** : Features sur échelles différentes

```
Age :     [20, 25, 30, 35, 40]           échelle: ~20
Salaire : [25000, 35000, 45000, 55000]   échelle: ~10000
```

**Impact sur algorithmes basés distance** :

Distance euclidienne :
$$d(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{\sum_{i=1}^{n} (x_{1i} - x_{2i})^2}$$

Sans normalisation :
$$d = \sqrt{(25-20)^2 + (35000-25000)^2} = \sqrt{25 + 10^8} \approx 10000$$

Salaire **domine** complètement la distance !

#### Techniques de Normalisation

**1. Standardisation (Z-score normalization)**

**Formule** :
$$z = \frac{x - \mu}{\sigma}$$

où $\mu$ = moyenne, $\sigma$ = écart-type

**Propriétés mathématiques** :
- $\mathbb{E}[z] = 0$
- $\text{Var}(z) = 1$
- Préserve la forme de la distribution
- Outliers conservés (mais moins influents)

**Distribution résultante** :
- Si X suit $\mathcal{N}(\mu, \sigma^2)$, alors Z suit $\mathcal{N}(0, 1)$

**Quand utiliser ?**
- ✅ Algorithmes supposant distribution normale (SVM, Logistic Regression)
- ✅ Présence d'outliers à conserver
- ✅ Distance euclidienne importante (KNN, K-means)

**2. Min-Max Scaling**

**Formule** :
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

**Propriétés** :
- $x' \in [0, 1]$
- Préserve relations originales
- **Très sensible aux outliers** : Si max est outlier, tous les autres compressés vers 0

**Variante : Scaling vers [a, b]**
$$x' = a + \frac{(x - \min(x))(b-a)}{\max(x) - \min(x)}$$

**Quand utiliser ?**
- ✅ Distribution bornée connue
- ✅ Neural Networks (activation functions)
- ✅ Valeurs absolues importantes
- ❌ Présence d'outliers

**3. Robust Scaling**

**Formule** :
$$x' = \frac{x - \text{median}(x)}{IQR}$$

où $IQR = Q_3 - Q_1$ (Interquartile Range)

**Propriétés** :
- ✅ **Robuste aux outliers** (utilise médiane et IQR)
- Ne garantit pas plage spécifique

**Quand utiliser ?**
- ✅ Présence de nombreux outliers
- ✅ Distribution très asymétrique

**4. Log Transform**

**Formule** :
$$x' = \log(x + c)$$

où $c$ est une constante (souvent 1)

**Effet** :
- Réduit l'asymétrie (skewness)
- Compresse grandes valeurs
- Étend petites valeurs

**Quand utiliser ?**
- ✅ Distribution log-normale
- ✅ Données avec grande plage (revenus, populations)
- ✅ Présence d'outliers à compresser

---

### D. Théorie de la Sélection de Features

#### Motivation Théorique

**Curse of Dimensionality** :

En haute dimension :
1. Les points deviennent **équidistants**
2. Volume de l'espace explose : $V = r^d$ (d dimensions)
3. Données deviennent **sparse**

**Conséquence** : Besoin exponentiel de données quand dimension augmente

**Solution** : Réduire nombre de features

#### Les 3 Familles d'Approches

**1. Filter Methods (Méthodes de Filtrage)**

**Principe** : Scorer chaque feature indépendamment du modèle

**Métriques utilisées** :

- **Corrélation de Pearson** : $r = \frac{\text{cov}(X,y)}{\sigma_X \sigma_y}$
  - Mesure relation linéaire
  - Entre -1 et 1

- **Chi-squared test** : $\chi^2 = \sum \frac{(O - E)^2}{E}$
  - Pour features catégorielles
  - Test d'indépendance

- **Mutual Information** : $I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$
  - Mesure dépendance (linéaire ou non)
  - ≥ 0, plus grand = plus dépendant

**Avantages** :
- ✅ Rapide (pas d'entraînement de modèle)
- ✅ Indépendant du modèle final

**Inconvénients** :
- ❌ Ignore interactions entre features
- ❌ Univarié (une feature à la fois)

**2. Wrapper Methods (Méthodes d'Enveloppe)**

**Principe** : Évaluer sous-ensembles de features selon **performance du modèle**

**Algorithmes** :

**Forward Selection** :
1. Commencer avec 0 features
2. Ajouter feature qui améliore le plus le modèle
3. Répéter jusqu'à condition d'arrêt

**Backward Elimination** :
1. Commencer avec toutes les features
2. Retirer feature qui dégrade le moins le modèle
3. Répéter jusqu'à condition d'arrêt

**Recursive Feature Elimination (RFE)** :
1. Entraîner modèle avec toutes features
2. Classer features par importance
3. Retirer features les moins importantes
4. Répéter

**Avantages** :
- ✅ Considère interactions entre features
- ✅ Optimise pour le modèle final

**Inconvénients** :
- ❌ Coûteux computationnellement : $O(2^n)$ possibilités
- ❌ Risque d'overfitting

**3. Embedded Methods (Méthodes Intégrées)**

**Principe** : Sélection de features **intégrée dans l'algorithme**

**Exemples** :

**Lasso (L1 Regularization)** :
$$\min_\beta \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

- Pénalité L1 pousse coefficients vers 0
- Effectue sélection automatique : $\beta_j = 0$ pour features non importantes

**Tree-based Feature Importance** :
- Random Forest : Importance = Réduction impurity moyenne
- XGBoost : Gain moyen par split

**Avantages** :
- ✅ Plus rapide que Wrapper
- ✅ Capture interactions
- ✅ Moins d'overfitting que Wrapper

---

### E. Théorie des Outliers

#### Définition Statistique

**Outlier** = Observation significativement différente des autres

**Définitions formelles** :

**1. Méthode IQR** :
- Outlier si $x < Q_1 - 1.5 \times IQR$ ou $x > Q_3 + 1.5 \times IQR$
- Basée sur boxplot de Tukey

**2. Méthode Z-score** :
- Outlier si $|z| > 3$ (ou 2.5)
- $z = \frac{x - \mu}{\sigma}$
- Suppose distribution normale

**3. Méthode Modified Z-score** :
- $M_i = \frac{0.6745(x_i - \tilde{x})}{MAD}$
- où $MAD = \text{median}(|x_i - \tilde{x}|)$
- Plus robuste que Z-score

#### Types d'Outliers

**1. Point Outliers** : Une valeur isolée extrême

**2. Contextual Outliers** : Extrême dans un contexte
- Exemple : 35°C normal en été, outlier en hiver

**3. Collective Outliers** : Groupe de points anormaux
- Exemple : Série temporelle avec pattern anormal

#### Décision : Garder ou Supprimer ?

**Garder si** :
- Outlier = phénomène d'intérêt (fraude, maladie rare)
- Outlier = variation naturelle
- Information précieuse

**Supprimer/Traiter si** :
- Erreur de mesure
- Erreur de saisie
- Corruption de données
- Outlier biaise le modèle

**Transformations alternatives** :
1. **Log transformation** : Compresse outliers
2. **Winsorizing** : Remplacer par percentile (1%, 99%)
3. **Capping** : Plafonner à valeur maximale acceptable

---

## 3. ACP (PCA) - Réduction Dimensionnalité

### 🎯 Définition

**PCA** = Principal Component Analysis = Analyse en Composantes Principales

**Objectif mathématique** : Projeter données de dimension $p$ vers dimension $k$ ($k < p$) en **maximisant la variance conservée**

### 📐 Fondements Mathématiques

#### Problème d'Optimisation

Soit $\mathbf{X}$ matrice de données $n \times p$ (n observations, p features)

**PCA cherche** directions $\mathbf{u}_1, \mathbf{u}_2, ..., \mathbf{u}_k$ telles que :

$$\mathbf{u}_1 = \arg\max_{\|\mathbf{u}\|=1} \text{Var}(\mathbf{X}\mathbf{u})$$

Sous contrainte : $\mathbf{u}_i \perp \mathbf{u}_j$ pour $i \neq j$ (orthogonaux)

**Interprétation** :
- $\mathbf{u}_1$ = direction de variance maximale (PC1)
- $\mathbf{u}_2$ = direction de variance max orthogonale à $\mathbf{u}_1$ (PC2)
- etc.

#### Solution : Décomposition en Valeurs Propres

**Étapes mathématiques** :

1. **Centrer les données** :
   $$\mathbf{X}_{centered} = \mathbf{X} - \bar{\mathbf{X}}$$

2. **Calculer matrice de covariance** :
   $$\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$$

3. **Décomposition spectrale** :
   $$\mathbf{C} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^T$$
   
   où :
   - $\mathbf{U}$ = matrice des vecteurs propres (directions PC)
   - $\mathbf{\Lambda}$ = matrice diagonale des valeurs propres $\lambda_i$

4. **Variance expliquée** :
   $$\text{Variance expliquée par PC}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$$

5. **Transformation** :
   $$\mathbf{Z} = \mathbf{X}\mathbf{U}_k$$
   
   où $\mathbf{U}_k$ = k premiers vecteurs propres

#### Propriétés Mathématiques

**1. Orthogonalité** :
$$\mathbf{u}_i^T \mathbf{u}_j = \begin{cases} 1 & \text{si } i = j \\ 0 & \text{si } i \neq j \end{cases}$$

**2. Variance totale conservée** :
$$\sum_{i=1}^{p} \lambda_i = \sum_{j=1}^{p} \text{Var}(X_j)$$

**3. Décorrélation** :
Les composantes principales sont **non corrélées** : $\text{Cov}(PC_i, PC_j) = 0$ pour $i \neq j$

**4. Maximisation de la variance** :
Les k premières PC capturent **le maximum de variance** possible avec k dimensions

---

### 🎯 Interprétation Géométrique

#### Rotation des Axes

PCA effectue une **rotation rigide** des axes de coordonnées

```
Avant PCA :         Après PCA :
   
   x₂ |                PC2 ↗
      |   •  •           |
      | •  •   •         |  ○ ○
      |•  •  •           | ○ ○ ○
   ───┼───────── x₁   ───┼────────── PC1
      |                  |
```

- Nouvelle base : Axes alignés avec directions de variance maximale
- Données plus "étalées" le long de PC1
- PC2 capture variance résiduelle

#### Réduction de Dimension

**Principe** : Projeter sur les k premières PC

```
3D → 2D :

z   PC2 ↗
↑    ↗
|   ●
|  ●  ●      →    ○
| ●  ●            ○  ○
└──────→ x       ○  ○  ──→ PC1
  y ↗            (perte PC3)
```

**Ce qu'on perd** : Variance capturée par PC3, PC4, ...
**Ce qu'on garde** : Maximum de variance dans k dimensions

---

### 📊 Outils d'Analyse PCA

#### 1. Scree Plot (Graphique des Éboulis)

**Définition** : Graphique des valeurs propres en fonction du numéro de composante

```
Variance
    ↑
    |  ●
    |    ●
    |      ● ──── "Coude"
    |        ●___●___●___●___
    └────────────────────→ PC
         1  2  3  4  5  6
```

**Interprétation** :
- **Méthode du coude** : Garder composantes avant le "coude"
- Coude = point où variance marginale devient faible

**Théorie** :
- Chercher point où $\lambda_k - \lambda_{k+1}$ devient petit
- PC après le coude = principalement du bruit

#### 2. Variance Expliquée Cumulée

**Formule** :
$$\text{Cumulative variance} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i}$$

**Règle pratique** :
- Garder k composantes telles que variance cumulée ≥ 80% (ou 90%, 95%)

**Exemple** :
```
PC1: 65%  →  Cumulée: 65%
PC2: 25%  →  Cumulée: 90%  ← Suffisant !
PC3: 7%   →  Cumulée: 97%
PC4: 2%   →  Cumulée: 99%
```

#### 3. Cercle de Corrélations

**Définition** : Visualisation des variables originales dans espace des PC

**Construction** :
- Axes : PC1 (horizontal), PC2 (vertical)
- Chaque variable = flèche depuis origine
- Longueur flèche = qualité de représentation

**Interprétation** :

**Angle entre flèches** :
- 0° : Variables fortement corrélées positivement
- 180° : Variables fortement corrélées négativement
- 90° : Variables non corrélées

**Distance au centre** :
- Proche du cercle : Variable bien représentée sur PC1-PC2
- Proche du centre : Variable mal représentée (variance sur autres PC)

**Exemple** :
```
    PC2
     ↑
     |  ↗ Revenu
     | ↗
     |→ Âge
────┼────→ PC1
     |
     ↓ Prix
```
Interprétation : Âge et Revenu corrélés positivement, Prix corrélé négativement

#### 4. Plan Factoriel (Biplot)

**Définition** : Projection des **individus** et **variables** simultanément sur PC1-PC2

**Éléments** :
- Points = individus (observations)
- Flèches = variables originales

**Interprétation** :

**Individus proches** :
- Profils similaires sur les variables

**Individu dans direction d'une flèche** :
- Valeur élevée pour cette variable

**Exemple** :
```
    PC2
     ↑
   Jeune ●
     |  ●     ↗ Diplôme
     |
────┼───●────→ PC1
     |    ●    → Salaire
   Âgé ●
```

Interprétation :
- PC1 oppose jeunes/âgés
- PC2 oppose diplômés/non diplômés
- Individus à droite : Salaire élevé

---

### ⚠️ Conditions d'Application

#### 1. Standardisation Obligatoire

**Problème sans standardisation** :

Si variables sur échelles différentes :
```
Âge:    [20, 30, 40]        variance ≈ 100
Salaire: [25000, 35000]      variance ≈ 25,000,000
```

PC1 sera **dominée** par Salaire !

**Solution** : Standardiser AVANT PCA
```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
pca = PCA().fit(X_scaled)
```

#### 2. Linéarité

**PCA suppose relations linéaires** entre variables

**Limitation** :
- Si relations non linéaires, PCA peut mal performer
- **Alternative** : Kernel PCA (utilise kernel trick)

#### 3. Perte d'Interprétabilité

**Composantes principales = combinaisons linéaires** :

$$PC_1 = 0.5 \cdot Age + 0.7 \cdot Revenu - 0.3 \cdot Prix + ...$$

**Problème** :
- PC1 n'a pas de sens métier direct
- Difficile d'expliquer résultats au business

**Trade-off** :
- ✅ Réduction dimensionnalité, meilleure performance
- ❌ Perte d'interprétabilité

---

### 🎯 Applications Théoriques

#### 1. Visualisation

**Problème** : Impossible de visualiser >3 dimensions

**Solution PCA** : Projeter sur 2D ou 3D
```
100 features → 2 PC → Scatter plot visualisable
```

#### 2. Accélération des Algorithmes

**Temps calcul** souvent $O(n \cdot p)$ ou $O(n \cdot p^2)$

Si $p \to k$ avec $k \ll p$ : **Gain de vitesse considérable**

Exemple :
- KNN avec 100 features : lent
- PCA → 10 features → KNN : rapide

#### 3. Réduction du Bruit

**Théorie** :
- Dernières PC = principalement du bruit
- En les supprimant → amélioration signal/bruit

#### 4. Multicolinéarité

**Problème** : Régression avec features corrélées → instabilité

**Solution** : PC sont **orthogonales** par construction
- Pas de multicolinéarité
- Régression sur PC stable

#### 5. Pré-traitement pour Clustering

**Avantage** :
- Réduit dimension → clustering plus rapide
- Réduit bruit → meilleurs clusters

---

## 4. K-Nearest Neighbors (KNN)

### 🎯 Définition

**KNN** = Algorithme **non paramétrique** de classification (ou régression) **supervisé** basé sur la **proximité**

**Principe** : *"Dis-moi qui sont tes voisins, je te dirai qui tu es"*

### 📐 Fondements Mathématiques

#### Algorithme

**Input** :
- Dataset d'entraînement $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$
- Nouveau point à classifier : $\mathbf{x}_{new}$
- Hyperparamètre : $k$ (nombre de voisins)

**Output** : Classe prédite $\hat{y}_{new}$

**Étapes** :

1. **Calculer distances** entre $\mathbf{x}_{new}$ et tous les $\mathbf{x}_i$ du training set

2. **Sélectionner k voisins** les plus proches
   $$\mathcal{N}_k(\mathbf{x}_{new}) = \text{k points } \mathbf{x}_i \text{ avec plus petites distances}$$

3. **Vote majoritaire** (classification)
   $$\hat{y}_{new} = \arg\max_{c} \sum_{i \in \mathcal{N}_k} \mathbb{1}(y_i = c)$$
   
   Ou **moyenne** (régression)
   $$\hat{y}_{new} = \frac{1}{k} \sum_{i \in \mathcal{N}_k} y_i$$

#### Métriques de Distance

**1. Distance Euclidienne** (la plus commune)
$$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2} = \|\mathbf{x} - \mathbf{x}'\|_2$$

**2. Distance de Manhattan** (L1)
$$d(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{p} |x_j - x'_j|$$

**3. Distance de Minkowski** (généralisation)
$$d(\mathbf{x}, \mathbf{x}') = \left(\sum_{j=1}^{p} |x_j - x'_j|^q\right)^{1/q}$$

- q=1 : Manhattan
- q=2 : Euclidienne
- q→∞ : Chebyshev

**4. Distance de Hamming** (variables catégorielles)
$$d(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{p} \mathbb{1}(x_j \neq x'_j)$$

---

### 🎯 Propriétés Théoriques

#### 1. Non-Paramétrique

**Définition** : Pas de phase d'entraînement, pas de paramètres à apprendre

**Lazy Learning** :
- Entraînement : Simplement **stocker** les données
- Prédiction : Calculs effectués **à la demande**

**Conséquence** :
- ✅ Flexible, s'adapte à des formes complexes
- ❌ Prédiction lente (doit calculer toutes distances)

#### 2. Frontière de Décision

**Théorie** : KNN crée frontières de décision **non linéaires**

```
k=1 :                k=5 :               k=15 :
Frontière            Frontière           Frontière
très irrégulière     lisse               très lisse

   ●●●●               ●●●●                ●●●●
  ●  ○○              ●  │○○              ●  │○○
 ●    ○○            ●   │ ○○            ●   │ ○○
●●    ○             ●   │  ○             ●   │  ○
```

**Observation** :
- k petit → Frontière complexe → Risque **overfitting**
- k grand → Frontière simple → Risque **underfitting**

#### 3. Théorème de Convergence

**Théorème de Cover-Hart (1967)** :

Quand $n \to \infty$, l'erreur de KNN-1 (k=1) est bornée :

$$\mathbb{P}(\text{erreur KNN-1}) \leq 2 \times \mathbb{P}(\text{erreur Bayes})$$

**Interprétation** :
- Bayes error = erreur minimale théorique
- KNN-1 avec données infinies : Au pire 2× erreur optimale
- **KNN est asymptotiquement bon** !

---

### ⚙️ Choix de k - Analyse Théorique

#### Biais-Variance Trade-off

**k petit (ex: k=1)** :
- **Biais faible** : Frontière flexible, suit données
- **Variance élevée** : Sensible au bruit → Overfitting
- Modèle **complexe**

**k grand (ex: k=n)** :
- **Biais élevé** : Frontière rigide
- **Variance faible** : Stable
- Modèle **simple** → Underfitting

**k optimal** : Équilibre entre biais et variance

#### Méthodes de Sélection de k

**1. Validation Croisée**

```python
for k in range(1, max_k):
    scores = cross_val_score(KNN(k), X, y, cv=5)
    mean_score[k] = scores.mean()

k_optimal = argmax(mean_score)
```

**2. Règle empirique**
$$k \approx \sqrt{n}$$

où n = taille dataset

**3. k impair pour classification binaire**
- Évite égalités dans le vote

---

### ⚠️ Sensibilité à l'Échelle

**Problème théorique fondamental** :

Distance euclidienne :
$$d = \sqrt{(x_1 - x'_1)^2 + (x_2 - x'_2)^2}$$

Si $x_1 \in [0, 100]$ et $x_2 \in [0, 1]$ :
$$d \approx \sqrt{(x_1 - x'_1)^2} \approx |x_1 - x'_1|$$

**Feature 1 domine complètement la distance !**

**Solution impérative** : **Standardisation**

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

Ainsi toutes features contribuent équitablement à la distance.

---

### 📊 Complexité Algorithmique

**Entraînement** : $O(1)$
- Juste stocker les données

**Prédiction naïve** : $O(n \cdot p \cdot m)$
- n = taille training set
- p = nombre de features
- m = nombre de prédictions

**Pour chaque prédiction** :
1. Calculer n distances : $O(n \cdot p)$
2. Trier pour trouver k plus proches : $O(n \log k)$

**Total** : $O(n \cdot p) + O(n \log k) = O(n \cdot p)$

**Problème** : Lent sur grands datasets !

**Solutions** :
1. **KD-Tree** : Structure arborescente, $O(\log n)$ en moyenne
2. **Ball Tree** : Alternative pour haute dimension
3. **Approximate NN** : Algorithmes type LSH (Locality Sensitive Hashing)

---

### 🎯 Avantages et Limitations

**Avantages** :
- ✅ Simple à comprendre et implémenter
- ✅ Pas d'hypothèse sur distribution des données
- ✅ Frontières décision non linéaires
- ✅ Peut faire classification et régression

**Limitations** :
- ❌ **Curse of dimensionality** : Performance se dégrade en haute dimension
- ❌ Prédiction lente sur grands datasets
- ❌ Sensible au bruit et outliers
- ❌ Nécessite standardisation
- ❌ Nécessite beaucoup de mémoire (stocker toutes données)

---

## 5. Support Vector Machine (SVM)

### 🎯 Définition

**SVM** = Algorithme **supervisé** de classification cherchant l'**hyperplan optimal** séparant les classes

**Objectif** : Maximiser la **marge** entre les classes

### 📐 Fondements Mathématiques

#### Cas Linéairement Séparable

**Hyperplan** en dimension p :
$$\mathbf{w}^T \mathbf{x} + b = 0$$

où :
- $\mathbf{w}$ = vecteur normal (perpendiculaire) à l'hyperplan
- $b$ = biais (intercept)

**Règle de décision** :
$$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b) = \begin{cases} +1 & \text{si } \mathbf{w}^T \mathbf{x} + b > 0 \\ -1 & \text{si } \mathbf{w}^T \mathbf{x} + b < 0 \end{cases}$$

#### Distance Point-Hyperplan

Distance d'un point $\mathbf{x}_i$ à l'hyperplan :

$$d(\mathbf{x}_i, H) = \frac{|\mathbf{w}^T \mathbf{x}_i + b|}{\|\mathbf{w}\|}$$

#### Marge

**Marge** = Distance minimale entre hyperplan et les points les plus proches

$$\text{marge} = \min_i \frac{|\mathbf{w}^T \mathbf{x}_i + b|}{\|\mathbf{w}\|}$$

**Support Vectors** = Points sur la marge (les plus proches de l'hyperplan)

#### Problème d'Optimisation (Hard Margin)

**Objectif** : Maximiser la marge

$$\max_{\mathbf{w}, b} \frac{1}{\|\mathbf{w}\|}$$

Équivalent à :

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$

**Sous contraintes** :
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i$$

**Interprétation** :
- Minimiser $\|\mathbf{w}\|$ ⟺ Maximiser marge
- Contraintes : Tous points correctement classés avec marge ≥ 1

**Solution** : Problème d'optimisation convexe (QPP - Quadratic Programming Problem)

---

### 📐 Formulation Duale (Lagrangien)

**Lagrangien** :
$$\mathcal{L}(\mathbf{w}, b, \alpha) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{n} \alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1]$$

où $\alpha_i \geq 0$ = multiplicateurs de Lagrange

**Conditions KKT** (Karush-Kuhn-Tucker) :
1. $\alpha_i \geq 0$
2. $y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1 \geq 0$
3. $\alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1] = 0$ (**Complementary slackness**)

**Théorie** : Point 3 implique :
- Si $\alpha_i > 0$ ⟹ $y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1$ ⟹ $\mathbf{x}_i$ est un **support vector**
- Si $\alpha_i = 0$ ⟹ $\mathbf{x}_i$ n'est pas sur la marge ⟹ Pas un support vector

**Formulation duale** :
$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

Sous contraintes :
- $\alpha_i \geq 0, \forall i$
- $\sum_{i=1}^{n} \alpha_i y_i = 0$

**Avantage du dual** :
- Dépend uniquement de **produits scalaires** $\mathbf{x}_i^T \mathbf{x}_j$
- Permet l'utilisation du **kernel trick** !

---

### 🔧 Soft Margin SVM

**Problème** : Données rarement linéairement séparables en pratique

**Solution** : Autoriser **violations de la marge** avec pénalité

**Variables de slack** : $\xi_i \geq 0$

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

**Sous contraintes** :
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \forall i$$

**Interprétation** :
- $\xi_i = 0$ : Point correctement classé, sur ou au-delà de la marge
- $0 < \xi_i < 1$ : Point dans la marge mais côté correct
- $\xi_i \geq 1$ : Point mal classé

**Hyperparamètre C** :
- **C grand** : Pénalité forte, peu de violations → **Hard margin**, risque overfitting
- **C petit** : Pénalité faible, beaucoup de violations → **Soft margin**, risque underfitting

**Trade-off** :
$$\text{Minimiser} : \underbrace{\|\mathbf{w}\|^2}_{\text{marge large}} + \underbrace{C \sum \xi_i}_{\text{erreurs faibles}}$$

---

### 🔮 Kernel Trick

**Problème** : Beaucoup de problèmes non linéairement séparables

**Idée** : Projeter données dans espace de dimension supérieure où elles deviennent séparables

$$\phi : \mathbb{R}^p \to \mathbb{R}^d, \quad d \gg p$$

**Problème** : Calcul en haute dimension coûteux !

**Solution : Kernel Trick**

Au lieu de calculer $\phi(\mathbf{x})$ explicitement, utiliser **fonction kernel** :

$$K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')$$

**Avantage** : Calcul produit scalaire en haute dimension **sans calculer $\phi$ explicitement** !

---

### 🎨 Types de Kernels

#### 1. Linear Kernel

$$K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$$

**Usage** : Données déjà linéairement séparables

**Équivalent à** : SVM linéaire standard

#### 2. Polynomial Kernel

$$K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + c)^d$$

où :
- $d$ = degré du polynôme
- $c$ = constante

**Exemple d = 2** :
Si $\mathbf{x} = (x_1, x_2)$ :

$$\phi(\mathbf{x}) = (x_1^2, \sqrt{2}x_1 x_2, x_2^2, \sqrt{2c}x_1, \sqrt{2c}x_2, c)$$

**Dimension** : $\binom{p+d}{d}$ (combinatoire)

**Usage** : Relations polynomiales entre features

#### 3. RBF Kernel (Radial Basis Function) ⭐

$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$

où $\gamma = \frac{1}{2\sigma^2}$ = paramètre de largeur

**Aussi appelé** : Gaussian Kernel

**Propriétés** :
- $K(\mathbf{x}, \mathbf{x}) = 1$
- $K(\mathbf{x}, \mathbf{x}') \to 0$ quand $\|\mathbf{x} - \mathbf{x}'\| \to \infty$
- Projette dans espace de **dimension infinie** !

**Hyperparamètre γ** :
- **γ grand** : Influence locale, frontière complexe → Risque overfitting
- **γ petit** : Influence globale, frontière lisse → Risque underfitting

**Usage** : **Le plus polyvalent**, bon point de départ

#### 4. Sigmoid Kernel

$$K(\mathbf{x}, \mathbf{x}') = \tanh(\alpha \mathbf{x}^T \mathbf{x}' + c)$$

**Équivalent à** : Réseau de neurones simple

**Problème** : Pas toujours défini positif (conditions Mercer)

---

### 🎯 Hyperparamètres SVM

#### 1. Paramètre C (Régularisation)

**Rôle** : Contrôle trade-off marge/erreurs

**C → ∞** (très grand) :
- Hard margin
- Aucune erreur tolérée
- Risque **overfitting**

**C → 0** (très petit) :
- Soft margin
- Beaucoup d'erreurs tolérées
- Risque **underfitting**

**Valeurs typiques** : 0.1, 1, 10, 100

#### 2. Paramètre γ (pour RBF)

**Formule** :
$$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$$

**Rôle** : Contrôle portée de l'influence d'un point

**γ grand** (ex: 10) :
- Influence locale forte
- Seuls voisins très proches comptent
- Frontière complexe, sinueuse
- Risque **overfitting**

**γ petit** (ex: 0.1) :
- Influence globale
- Beaucoup de points influents
- Frontière lisse
- Risque **underfitting**

**Relation avec σ** :
$$\gamma = \frac{1}{2\sigma^2}$$

**Règle heuristique** :
$$\gamma_{\text{default}} = \frac{1}{p}$$

où p = nombre de features

#### 3. Degré d (pour Polynomial)

**Rôle** : Complexité des relations polynomiales

**d = 1** : Linéaire
**d = 2** : Quadratique
**d = 3** : Cubique

**Augmenter d** :
- ✅ Plus flexible
- ❌ Overfitting
- ❌ Plus coûteux computationnellement

---

### 📊 Complexité Algorithmique

**Entraînement** :
- **Meilleur cas** : $O(n^2 \cdot p)$
- **Pire cas** : $O(n^3)$

où n = nombre d'exemples, p = nombre de features

**Prédiction** : $O(n_{sv} \cdot p)$

où $n_{sv}$ = nombre de support vectors

**Théorie** : Nombre de SV généralement $\ll n$

**Problème** : Lent sur très grands datasets (n > 100,000)

**Solutions** :
- **SGD-SVM** : Stochastic Gradient Descent SVM
- **Linear SVM optimisé** : liblinear

---

### 🎯 Propriétés Théoriques

#### 1. Maximum Marge

**Théorie** : Maximiser marge → Meilleure généralisation

**Intuition** : Grande marge = robustesse aux perturbations

**Preuve (intuitive)** :
- Petite marge: Point test légèrement bruité peut changer de côté → erreur
- Grande marge: Plus de "sécurité", robuste au bruit

#### 2. Support Vectors

**Propriété clé** : Seuls les SV déterminent l'hyperplan

**Conséquence** :
- Points loin de la marge sont **ignorés**
- Modèle **sparse** : Dépend de peu de points
- ✅ Robuste aux outliers loin de la marge
- ❌ Sensible aux outliers sur la marge

#### 3. Marges et VC-Dimension

**Théorie de Vapnik** :

VC-dimension (capacité du modèle) liée à la marge :

$$h \approx \min\left(\frac{R^2}{\rho^2}, p\right) + 1$$

où :
- R = rayon de la sphère contenant données
- ρ = marge
- p = dimension

**Interprétation** : Grande marge → Petite VC-dimension → Bonne généralisation

---

### ⚠️ Conditions d'Application

#### 1. Standardisation Obligatoire

**Raison** : SVM utilise distances (produits scalaires)

Sans standardisation :
- Features à grande échelle dominent
- Marge biaisée

#### 2. Choix du Kernel

**Heuristique** :

1. Commencer avec **RBF kernel** (le plus flexible)
2. Si surapprentissage : Essayer **linear kernel**
3. Si problème spécifique : Polynomial ou custom kernel

**Règle** : Si $n < p$ (peu d'exemples, beaucoup de features) :
- Linear kernel souvent suffisant
- RBF risque overfitting

---

### 🎯 Avantages et Limitations

**Avantages** :
- ✅ Efficace en haute dimension
- ✅ Frontières décision complexes (via kernels)
- ✅ Théoriquement fondé (théorie PAC)
- ✅ Robuste (maximum marge)
- ✅ Modèle sparse (support vectors)

**Limitations** :
- ❌ Lent sur grands datasets (n > 100K)
- ❌ Sensible au choix hyperparamètres (C, γ)
- ❌ Sensible au déséquilibre des classes
- ❌ Ne donne pas probabilités directes
- ❌ Difficile à interpréter (surtout avec kernels)

---

## 6. K-means Clustering

### 🎯 Définition

**K-means** = Algorithme de **clustering non supervisé** partitionnant n observations en k clusters par minimisation de la variance intra-cluster

**Objectif** : Regrouper données similaires ensemble

### 📐 Fondements Mathématiques

#### Problème d'Optimisation

**Input** :
- Dataset $\mathbf{X} = \{\mathbf{x}_1, ..., \mathbf{x}_n\}$ avec $\mathbf{x}_i \in \mathbb{R}^p$
- Nombre de clusters $k$

**Output** :
- k centroïdes $\{\mu_1, ..., \mu_k\}$
- Assignations $C = \{C_1, ..., C_k\}$ où $C_j$ = ensemble points du cluster j

**Fonction objectif (Inertie)** :

$$J = \sum_{j=1}^{k} \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \mu_j\|^2$$

**Objectif** : $\min_{C, \mu} J$

**Interprétation** :
- Minimiser somme des carrés des distances entre points et leur centroïde
- = Minimiser **variance intra-cluster**
- = Maximiser **compacité** des clusters

---

### 🔄 Algorithme K-means (Lloyd's Algorithm)

**Initialisation** :
1. Choisir k centroïdes initiaux $\mu_1^{(0)}, ..., \mu_k^{(0)}$

**Itération** (jusqu'à convergence) :

**Étape 1 : Assignment Step**
Assigner chaque point au centroïde le plus proche :

$$C_j^{(t)} = \{\mathbf{x}_i : \|\mathbf{x}_i - \mu_j^{(t)}\| \leq \|\mathbf{x}_i - \mu_{j'}^{(t)}\|, \forall j'\}$$

**Étape 2 : Update Step**
Recalculer centroïdes comme moyenne des points assignés :

$$\mu_j^{(t+1)} = \frac{1}{|C_j^{(t)}|} \sum_{\mathbf{x}_i \in C_j^{(t)}} \mathbf{x}_i$$

**Convergence** : S'arrêter quand :
- Inertie ne change plus (ou change < ε)
- Centroïdes ne bougent plus
- Max itérations atteint

---

### 📊 Propriétés de Convergence

#### Théorème de Convergence

**Propriété fondamentale** : K-means **converge toujours** vers un minimum

**Preuve (esquisse)** :

1. **Assignment step** : $J^{(t+1)} \leq J^{(t)}$ (assignations optimales)
2. **Update step** : $J^{(t+1)} \leq J^{(t)}$ (moyennes minimisent variance)
3. J est **décroissante** et **bornée** inférieurement (≥ 0)
4. Donc J **converge**

**MAIS** :
- ❌ Convergence vers **minimum local**, pas forcément global
- ❌ Résultat dépend de l'**initialisation**

#### Complexité

**Par itération** : $O(n \cdot k \cdot p)$
- n = nombre de points
- k = nombre de clusters
- p = dimension

**Nombre d'itérations** : Typiquement < 100

**Complexité totale** : $O(t \cdot n \cdot k \cdot p)$

où t = nombre d'itérations

---

### 🎲 Méthodes d'Initialisation

#### 1. Random Initialization (Aléatoire)

**Méthode** : Choisir k points au hasard comme centroïdes initiaux

**Problème** : Très sensible à l'initialisation

**Solution** : Exécuter algorithme **plusieurs fois** (n_init) et garder meilleur résultat

#### 2. K-means++ ⭐ (RECOMMANDÉ)

**Développé par Arthur & Vassilvitskii (2007)**

**Algorithme** :

1. Choisir premier centroïde **aléatoirement** parmi les points

2. Pour chaque centroïde suivant :
   - Calculer distance $D(\mathbf{x})$ de chaque point au centroïde le plus proche
   - Choisir prochain centroïde avec probabilité $\propto D(\mathbf{x})^2$

3. Répéter jusqu'à k centroïdes

**Intuition** : Espacer les centroïdes initiaux

**Propriété théorique** :

$$\mathbb{E}[J_{K\text{-means++}}] \leq 8(\ln k + 2) \cdot J_{OPT}$$

**Interprétation** : K-means++ garantit solution à facteur $O(\log k)$ de l'optimal

**Avantage** :
- ✅ Meilleure initialisation
- ✅ Convergence plus rapide
- ✅ Moins sensible aux minima locaux

---

### 🎯 Choix du Nombre de Clusters k

#### Problème

K-means nécessite de **spécifier k** à l'avance

**Question** : Comment choisir k optimal ?

#### Méthode 1 : Elbow Method (Méthode du Coude) ⭐

**Principe** : Tracer inertie en fonction de k

```
Inertie
    ↑
    |●
    | ●
    |  ●
    |    ●  ← Coude ici (k=3)
    |      ●___●___●___
    └─────────────────→ k
         1 2 3 4 5 6 7
```

**Interprétation** :
- Inertie décroît toujours avec k (plus de clusters = moins de variance)
- Chercher "coude" = point où gain marginal devient faible

**Formule de gain marginal** :
$$\Delta_k = J_{k-1} - J_k$$

Chercher k où $\Delta_{k+1} \ll \Delta_k$

**Problème** : Coude pas toujours clair !

#### Méthode 2 : Silhouette Score ⭐

**Pour chaque point $i$** :

**a(i)** = Distance moyenne aux points du **même cluster** :
$$a(i) = \frac{1}{|C_j| - 1} \sum_{\mathbf{x}_k \in C_j, k \neq i} d(\mathbf{x}_i, \mathbf{x}_k)$$

**b(i)** = Distance moyenne au cluster **voisin le plus proche** :
$$b(i) = \min_{j \neq j_i} \frac{1}{|C_j|} \sum_{\mathbf{x}_k \in C_j} d(\mathbf{x}_i, \mathbf{x}_k)$$

**Silhouette de $i$** :
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Silhouette moyenne** :
$$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

**Interprétation** :
- $s(i) \approx 1$ : Point bien clustérisé (loin des autres clusters)
- $s(i) \approx 0$ : Point entre deux clusters
- $s(i) \approx -1$ : Point mal clustérisé (plus proche d'un autre cluster)

**Choisir k** : Maximiser $\bar{s}$

**Plage** : $s \in [-1, 1]$
- $\bar{s} > 0.7$ : Structure forte
- $\bar{s} > 0.5$ : Structure raisonnable
- $\bar{s} < 0.25$ : Pas de structure

#### Méthode 3 : Gap Statistic

**Principe** : Comparer inertie aux données **aléatoires**

$$\text{Gap}(k) = \mathbb{E}[\log(W_k)]_{\text{random}} - \log(W_k)_{\text{data}}$$

où $W_k$ = inertie totale

**Choisir k** : Plus petit k tel que :
$$\text{Gap}(k) \geq \text{Gap}(k+1) - s_{k+1}$$

**Interprétation** : k où clustering dépasse significativement le hasard

---

### 📐 Hypothèses de K-means

#### 1. Clusters Sphériques

**Hypothèse** : K-means suppose clusters de forme **sphérique** (variance isotrope)

**Problème si** :
- Clusters elliptiques
- Clusters de formes arbitraires

**Exemple où K-means échoue** :
```
Clusters en forme de demi-lunes :

    ●●●●●
   ●     ●●●
  ●         ●●
 ●            ●
●              ●

K-means divisera horizontalement (mauvais)
```

#### 2. Tailles Similaires

**K-means favorise** clusters de tailles équilibrées

**Problème** : Si un cluster beaucoup plus grand, peut être divisé

#### 3. Densités Similaires

**K-means suppose** variances similaires

**Si variances différentes** : Clusters denses sur-divisés

---

### ⚠️ Impact des Outliers

**Problème théorique** : Centroïdes = **moyennes**

**Conséquence** : Très sensible aux outliers

**Exemple** :
```
Cluster :  1, 2, 3, 4, 5  → moyenne = 3
Avec outlier : 1, 2, 3, 4, 5, 100 → moyenne = 19.17

Centroïde déplacé vers outlier !
```

**Solutions** :
1. **Détecter et supprimer** outliers avant clustering
2. **K-medoids** (PAM) : Utilise médiane au lieu de moyenne
3. **DBSCAN** : Robuste aux outliers

---

### ⚠️ Sensibilité à l'Échelle

**Théorie** : Distance euclidienne

$$d(\mathbf{x}, \mu) = \sqrt{\sum_{j=1}^{p} (x_j - \mu_j)^2}$$

**Si features sur échelles différentes** : Features grande échelle dominent

**Solution impérative** : **Standardisation**

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

---

### 🎯 Variantes de K-means

#### 1. Mini-Batch K-means

**Pour** : Très grands datasets

**Principe** : Utiliser mini-batches aléatoires au lieu de tous les points

**Avantages** :
- ✅ Beaucoup plus rapide
- ✅ Peut traiter datasets qui ne tiennent pas en mémoire

**Inconvénients** :
- ❌ Qualité légèrement inférieure

#### 2. K-medoids (PAM - Partitioning Around Medoids)

**Différence** : Centroïdes = **points réels** du dataset (medoids), pas moyennes

**Avantage** :
- ✅ Robuste aux outliers

**Inconvénient** :
- ❌ Plus coûteux : $O(n^2)$

#### 3. Fuzzy C-means

**Différence** : Assignations **probabilistes**

Chaque point appartient à **tous les clusters** avec degrés d'appartenance

$$\sum_{j=1}^{k} u_{ij} = 1$$

où $u_{ij}$ = degré d'appartenance de $i$ au cluster $j$

---

### 🎯 Avantages et Limitations

**Avantages** :
- ✅ Simple et intuitif
- ✅ Rapide : $O(n \cdot k \cdot p)$
- ✅ Scalable à grands datasets
- ✅ Convergence garantie

**Limitations** :
- ❌ Nécessite spécifier k
- ❌ Sensible à l'initialisation (minima locaux)
- ❌ Suppose clusters sphériques, tailles similaires
- ❌ Sensible aux outliers
- ❌ Sensible à l'échelle (nécessite standardisation)
- ❌ Mal adapté aux formes complexes

---

## 7. DBSCAN Clustering

### 🎯 Définition

**DBSCAN** = **D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise

Algorithme de clustering **non supervisé basé sur la densité**, développé par Ester et al. (1996)

**Principe** : Clusters = régions de **haute densité** séparées par régions de **basse densité**

### 📐 Concepts Fondamentaux

#### Paramètres

**1. ε (epsilon)** : Rayon de voisinage
**2. MinPts** : Nombre minimum de points dans voisinage

**Notation** :
- $N_\varepsilon(\mathbf{x})$ = Voisinage de $\mathbf{x}$ de rayon ε

$$N_\varepsilon(\mathbf{x}) = \{\mathbf{x}' \in \mathbf{X} : d(\mathbf{x}, \mathbf{x}') \leq \varepsilon\}$$

#### Types de Points

**1. Core Point (Point Central)**

Point $\mathbf{x}$ tel que $|N_\varepsilon(\mathbf{x})| \geq \text{MinPts}$

**Interprétation** : Point dans région dense

**2. Border Point (Point Frontière)**

Point $\mathbf{x}$ tel que :
- $|N_\varepsilon(\mathbf{x})| < \text{MinPts}$ (pas assez dense)
- **MAIS** $\mathbf{x} \in N_\varepsilon(\mathbf{x}_{core})$ pour un point central $\mathbf{x}_{core}$

**Interprétation** : Point à la périphérie d'un cluster

**3. Noise Point (Point de Bruit)**

Point qui n'est **ni central ni frontière**

**Interprétation** : Outlier, pas dans un cluster

**Label** : -1

#### Relations de Densité

**Directly Density-Reachable** (Directement Atteignable en Densité) :

$\mathbf{x}'$ est directement atteignable depuis $\mathbf{x}$ si :
1. $\mathbf{x}'  \in N_\varepsilon(\mathbf{x})$
2. $\mathbf{x}$ est un point central

**Density-Reachable** (Atteignable en Densité) :

$\mathbf{x}'$ est atteignable depuis $\mathbf{x}$ s'il existe une chaîne :
$$\mathbf{x} = \mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_m = \mathbf{x}'$$

où chaque $\mathbf{p}_{i+1}$ est directement atteignable depuis $\mathbf{p}_i$

**Density-Connected** (Connecté en Densité) :

$\mathbf{x}$ et $\mathbf{x}'$ sont connectés s'il existe $\mathbf{p}$ tel que :
- $\mathbf{x}$ est atteignable depuis $\mathbf{p}$
- $\mathbf{x}'$ est atteignable depuis $\mathbf{p}$

---

### 🔄 Algorithme DBSCAN

**Input** :
- Dataset $\mathbf{X}$
- ε (rayon)
- MinPts (seuil densité)

**Output** :
- Labels des clusters (0, 1, 2, ..., -1 pour bruit)

**Algorithme** :

```
1. Marquer tous points comme "non visités"
2. Initialiser cluster_id = 0

3. Pour chaque point p non visité :
   a. Marquer p comme visité
   b. Trouver N_ε(p) = voisins de p dans rayon ε
   
   c. Si |N_ε(p)| < MinPts :
      - Marquer p comme BRUIT (temporairement)
   
   d. Sinon (p est core point) :
      - cluster_id += 1
      - Créer nouveau cluster C = {p}
      - Initialiser queue Q = N_ε(p)
      
      e. Tant que Q non vide :
         - Retirer point q de Q
         - Si q non visité :
           * Marquer q comme visité
           * Trouver N_ε(q)
           * Si |N_ε(q)| ≥ MinPts :
             · Ajouter N_ε(q) à Q (q est aussi core)
         - Si q pas encore dans un cluster :
           * Ajouter q à C

4. Retourner labels
```

**Propriétés de l'algorithme** :
- Points frontières assignés au **premier cluster** qui les atteint
- Un point frontière peut théoriquement appartenir à plusieurs clusters (ambigu)

---

### 📊 Complexité Algorithmique

**Sans optimisation** : $O(n^2)$
- Pour chaque point : calculer distances à tous les autres

**Avec structure spatiale** : $O(n \log n)$
- **KD-Tree** ou **Ball Tree** pour recherche de voisins efficace

**Espace mémoire** : $O(n)$

**Facteurs influençant vitesse** :
- Dimension des données
- Densité des données
- Choix de ε et MinPts

---

### 🎯 Choix des Paramètres

#### 1. Choix de MinPts

**Règle heuristique** :
$$\text{MinPts} \geq p + 1$$

où $p$ = dimension des données

**Justification** : Au moins $p+1$ points pour définir une région dense en dimension $p$

**Valeurs typiques** :
- 2D : MinPts = 4
- 3D : MinPts = 5
- p-D : MinPts = $2 \times p$

**Règle simple** : MinPts = 4 ou 5 fonctionne souvent bien

**Effet de MinPts** :
- MinPts **trop petit** : Beaucoup de petits clusters, bruit classé comme clusters
- MinPts **trop grand** : Beaucoup de points marqués comme bruit

#### 2. Choix de ε (CRITIQUE) ⭐

**Méthode : k-distance plot**

**Principe** : Tracer distance au k-ième voisin pour chaque point

**Algorithme** :

1. Pour chaque point, calculer distance au MinPts-ième voisin
2. Trier ces distances par ordre croissant
3. Tracer le graphique
4. Chercher le **"coude"** (point d'inflexion)

```
Distance
au k-ème
voisin
    ↑
    |
    |              ●●●●●●●●●
    |          ●●●● (bruit)
    |       ●●●
    |    ●●●  ← COUDE ici !
    | ●●●       ε optimal ≈ 0.5
    |●●
    └─────────────────────→
         Points (triés)
```

**Interprétation** :
- **Avant le coude** : Points dans clusters (petites distances)
- **Après le coude** : Points de bruit (grandes distances)
- **ε optimal** ≈ Distance au coude

**Code théorique** :
```python
from sklearn.neighbors import NearestNeighbors

# k = MinPts
nbrs = NearestNeighbors(n_neighbors=MinPts)
nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)

# Distance au k-ième voisin (dernière colonne)
k_distances = np.sort(distances[:, -1])

# Tracer
plt.plot(k_distances)
plt.ylabel(f'{MinPts}-distance')
plt.xlabel('Points triés')
plt.show()

# Chercher coude visuellement
```

**Méthode automatique du coude** (Kneedle algorithm) :
Trouve coude par calcul de courbure maximale

#### Effet de ε

**ε trop petit** :
- Beaucoup de points marqués comme bruit
- Clusters fragmentés en petits morceaux

**ε trop grand** :
- Clusters fusionnent
- Peu ou pas de bruit détecté
- Perd structure fine

**Trade-off** : ε contrôle granularité du clustering

---

### 📐 Propriétés Théoriques

#### 1. Détection Automatique du Nombre de Clusters

**Avantage majeur** : Pas besoin de spécifier k !

Nombre de clusters **émergent** de la structure de densité

#### 2. Formes Arbitraires

**DBSCAN peut trouver** clusters de formes :
- Non convexes
- Spirales
- Demi-lunes
- Anneaux
- Formes complexes

**Exemple** :
```
K-means échoue :        DBSCAN réussit :

  ●●●●●                   ●●●●● (Cluster 1)
 ●    ●●                 ●    ●●
●      ●●       →       ●      ●● (Cluster 2)
 ●    ●                  ●    ●
  ●●●●                   ●●●●

K-means: 2 clusters      DBSCAN: 2 clusters
verticaux (mauvais)      en demi-lunes (correct)
```

#### 3. Robustesse aux Outliers

**Points de bruit** explicitement identifiés (label -1)

**Avantage** :
- ✅ Outliers ne biaisent pas les clusters
- ✅ Détection automatique d'anomalies

**Comparaison avec K-means** :
- K-means : Outliers **déplacent centroïdes**
- DBSCAN : Outliers **ignorés** (marqués comme bruit)

#### 4. Déterminisme (avec précaution)

**DBSCAN est déterministe** pour les core points

**Non-déterminisme** pour les border points :
- Peuvent être assignés à différents clusters selon l'ordre de traitement
- En pratique : impact négligeable

---

### ⚠️ Limitations Théoriques

#### 1. Densités Variables

**Problème majeur** : DBSCAN suppose **densité relativement uniforme**

**Si densités très différentes** :
- ε pour cluster dense → trop petit pour cluster sparse
- ε pour cluster sparse → trop grand pour cluster dense

**Exemple** :
```
Cluster dense :     Cluster sparse :
●●●●●●              ●  ●    ●
●●●●●●              ●     ●
●●●●●●                 ●    ●

Un seul ε ne marche pas pour les deux !
```

**Solutions** :
- **HDBSCAN** (Hierarchical DBSCAN) : Gère densités variables
- **OPTICS** : Variante de DBSCAN, hiérarchique

#### 2. Haute Dimension

**Curse of Dimensionality** :

En haute dimension :
- Notion de densité devient **ambiguë**
- Tous points deviennent équidistants
- Rayon ε difficile à choisir

**Règle pratique** : DBSCAN marche bien si $p \leq 20$

#### 3. Sensibilité aux Paramètres

**ε et MinPts** influencent fortement le résultat

**Problème** : Pas de "bon" ε universel

**Nécessite** : Analyse exploratoire pour chaque dataset

---

### 🆚 DBSCAN vs K-means

| Critère | K-means | DBSCAN |
|---------|---------|--------|
| **Nombre clusters** | À spécifier | Automatique |
| **Formes clusters** | Sphériques | Arbitraires |
| **Outliers** | Sensible | Robuste (détecte bruit) |
| **Densités** | Uniforme | Uniforme (limitation) |
| **Paramètres** | k | ε, MinPts |
| **Complexité** | $O(n \cdot k \cdot p)$ | $O(n \log n)$ optimisé |
| **Scalabilité** | Excellente | Bonne |
| **Déterminisme** | Aléatoire (init) | Déterministe (core) |

---

### 🎯 Quand Utiliser DBSCAN ?

**Utiliser DBSCAN si** :
- ✅ Clusters de **formes complexes** (non sphériques)
- ✅ Nombre de clusters **inconnu**
- ✅ Présence d'**outliers** à détecter
- ✅ Densités **relativement uniformes**
- ✅ Dimension **modérée** (< 20)

**Utiliser K-means si** :
- ✅ Clusters **sphériques**
- ✅ Nombre de clusters **connu**
- ✅ **Rapidité** primordiale (grands datasets)
- ✅ Densités **similaires**

---

### 🎯 Variantes de DBSCAN

#### 1. HDBSCAN (Hierarchical DBSCAN)

**Amélioration** : Gère **densités variables**

**Principe** : Construit hiérarchie de clusters sur différents ε

**Avantage** :
- ✅ Pas besoin de choisir ε
- ✅ Clusters de densités différentes

#### 2. OPTICS

**Ordering Points To Identify Clustering Structure**

**Similaire à** : DBSCAN hiérarchique

**Output** : Ordre des points + **reachability distance**

**Avantage** : Visualisation de la structure de clustering

#### 3. ST-DBSCAN

**Spatio-Temporal DBSCAN**

**Pour** : Données avec composantes spatiales ET temporelles

**Exemple** : Trajectoires GPS

---

## 8. Concepts Transversaux

### 📊 Évaluation des Modèles

#### Métriques de Classification

**1. Accuracy (Exactitude)**

$$\text{Accuracy} = \frac{\text{Nombre de prédictions correctes}}{\text{Nombre total de prédictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Limite** : Trompeuse si classes déséquilibrées

**Exemple** :
- 95% emails normaux, 5% spam
- Modèle qui prédit toujours "normal" : 95% accuracy !
- Mais rate TOUS les spams !

**2. Matrice de Confusion**

```
                 Prédit
              Pos    Neg
Réel  Pos     TP     FN
      Neg     FP     TN
```

- **TP** (True Positive) : Correctement prédit positif
- **TN** (True Negative) : Correctement prédit négatif
- **FP** (False Positive) : Faux positif (erreur type I)
- **FN** (False Negative) : Faux négatif (erreur type II)

**3. Precision (Précision)**

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interprétation** : Parmi les prédictions positives, quelle proportion est correcte ?

**Quand importante ?** : Coût élevé des faux positifs
- Exemple : Diagnostic cancer (éviter faux positifs anxiogènes)

**4. Recall (Rappel / Sensibilité)**

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interprétation** : Parmi les vrais positifs, quelle proportion est détectée ?

**Quand important ?** : Coût élevé des faux négatifs
- Exemple : Détection fraude (ne pas rater de fraudes)

**5. F1-Score** (moyenne harmonique)

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Avantage** : Équilibre precision et recall

**F1 élevé** : Bon équilibre des deux métriques

#### Métriques de Clustering

**1. Inertie (Somme des Carrés Intra-Cluster)**

$$\text{Inertia} = \sum_{j=1}^{k} \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \mu_j\|^2$$

**Interprétation** : Compacité des clusters (plus petit = mieux)

**Limite** : Décroît toujours avec k

**2. Silhouette Score**

Déjà décrit en détail dans section K-means

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$

**Plage d'évaluation** :
- > 0.7 : Excellent
- 0.5-0.7 : Bon
- 0.25-0.5 : Faible
- < 0.25 : Pas de structure

**Avantage** : Fonctionne sans labels (non supervisé)

---

### 🎯 Validation et Généralisation

#### 1. Train/Test Split

**Principe** : Séparer données en train (80%) et test (20%)

**Train** : Entraîner modèle
**Test** : Évaluer généralisation

**RÈGLE D'OR** : Ne **JAMAIS** entraîner sur test !

**Pourquoi ?** : Test simule données futures inconnues

#### 2. Validation Croisée (Cross-Validation)

**K-Fold CV** :

1. Diviser données en K folds (ex: K=5)
2. Pour chaque fold i :
   - Train sur K-1 folds
   - Test sur fold i
3. Moyenne des K scores

**Avantage** :
- ✅ Utilise toutes données pour train ET test
- ✅ Estime mieux la variance du modèle
- ✅ Réduit effet de split aléatoire

**Coût** : K fois plus lent

**Variantes** :
- **Stratified K-Fold** : Préserve proportions des classes
- **Leave-One-Out (LOO)** : K = n (un point test à chaque fois)

#### 3. Stratification

**Pour** : Classes déséquilibrées

**Principe** : Respecter proportions des classes dans train/test

**Exemple** :
```
Dataset : 90% classe A, 10% classe B
→ Train : 90% A, 10% B
→ Test : 90% A, 10% B
```

---

### 📈 Biais et Variance

#### Décomposition de l'Erreur

**Erreur totale** = Biais² + Variance + Bruit irréductible

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

**Biais** :
- Erreur due à **hypothèses simplificatrices**
- Modèle **trop simple**
- Underfitting

**Variance** :
- Sensibilité aux **fluctuations** des données d'entraînement
- Modèle **trop complexe**
- Overfitting

**Bruit** : Erreur irréductible dans les données

#### Trade-off Biais-Variance

```
Erreur
    ↑
    |    Variance
    |       /‾‾\
    |      /    \___
    | ___/    Erreur totale
    |/
    |\___       Biais²
    |    \____
    └──────────────→ Complexité modèle
```

**Objectif** : Trouver compromis optimal

---

### 📊 Standardisation : Théorie Complète

#### Pourquoi Standardiser ?

**1. Échelles différentes**

Features sur échelles différentes → Domination

**2. Algorithmes basés distance**

KNN, SVM, K-means, DBSCAN, PCA : Tous utilisent distances

**3. Gradient Descent**

Convergence plus rapide avec features standardisées

#### StandardScaler - Détails Mathématiques

**Transformation** :

$$z_j = \frac{x_j - \bar{x}_j}{s_j}$$

où :
- $\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}$ (moyenne)
- $s_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_{ij} - \bar{x}_j)^2}$ (écart-type)

**Propriétés résultantes** :
- $\mathbb{E}[z_j] = 0$
- $\text{Std}(z_j) = 1$

**Inverse** :
$$x_j = z_j \cdot s_j + \bar{x}_j$$

#### RÈGLE CRITIQUE : fit_transform vs transform

**Sur données d'entraînement** :
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
→ Calcule $\bar{x}$ et $s$ sur X_train

**Sur données de test** :
```python
X_test_scaled = scaler.transform(X_test)
```
→ Utilise mêmes $\bar{x}$ et $s$ calculés sur train

**POURQUOI ?**
- Test doit simuler données futures
- Futures données n'influencent pas les paramètres du modèle
- **Sinon** : Data leakage !

---

### 🎯 Résumé des Applications Pratiques

| Problème | Type | Algorithme Recommandé | Pourquoi |
|----------|------|----------------------|----------|
| Email spam/non spam | Classification binaire | SVM (RBF) ou Logistic Regression | Haute dimension, bonne séparation |
| Détection fraude | Classification déséquilibrée | SVM, Random Forest | Classes déséquilibrées, importance rappel |
| Reconnaissance images | Classification | Deep Learning (CNN) | Haute dimension, patterns complexes |
| Prédiction prix maison | Régression | Régression Linéaire, Random Forest | Relations linéaires/non-linéaires |
| Segmentation clients | Clustering | K-means | Clusters sphériques, k connu |
| Détection anomalies réseau | Clustering + outliers | DBSCAN | Détection bruit, formes complexes |
| Réduction dimension visualisation | Visualisation | PCA | Réduction à 2D/3D |
| Système de recommandation | Collaborative Filtering | Matrix Factorization, ALS | Données sparse |

---

## 📌 Checklist Finale des Concepts

### ✅ CRISP-DM
- [ ] Je comprends les 6 phases
- [ ] Je sais que c'est itératif (pas linéaire)
- [ ] Je connais l'objectif de chaque phase

### ✅ Preprocessing
- [ ] Je connais les 3 types de NA (MCAR, MAR, MNAR)
- [ ] Je sais quand imputer par moyenne vs médiane
- [ ] Je comprends Label vs One-Hot Encoding
- [ ] Je sais pourquoi encoder AVANT scaler
- [ ] Je comprends StandardScaler (formule et usage)
- [ ] Je sais utiliser fit_transform vs transform

### ✅ PCA
- [ ] Je comprends l'objectif (réduction dimensionnalité)
- [ ] Je sais que PCA = maximisation variance
- [ ] Je comprends variance expliquée
- [ ] Je sais interpréter scree plot et cercle corrélations
- [ ] Je sais pourquoi standardiser AVANT PCA

### ✅ KNN
- [ ] Je comprends le principe (vote des voisins)
- [ ] Je sais que c'est lazy learning (non paramétrique)
- [ ] Je comprends impact de k (petit vs grand)
- [ ] Je sais que KNN est sensible à l'échelle
- [ ] Je connais complexité : $O(n \cdot p)$ par prédiction

### ✅ SVM
- [ ] Je comprends hyperplan et marge
- [ ] Je connais différence Hard vs Soft margin (rôle de C)
- [ ] Je connais les kernels (linear, RBF, polynomial)
- [ ] Je comprends rôle de γ (local vs global)
- [ ] Je sais que SVM nécessite standardisation

### ✅ K-means
- [ ] Je comprends objectif (minimiser inertie)
- [ ] Je connais l'algorithme (assignment + update)
- [ ] Je sais qu'il converge vers minimum local
- [ ] Je connais méthodes choix k (elbow, silhouette)
- [ ] Je sais que K-means suppose clusters sphériques
- [ ] Je comprends K-means++ (meilleure initialisation)
- [ ] Je sais que K-means sensible aux outliers

### ✅ DBSCAN
- [ ] Je comprends concept de densité
- [ ] Je connais les 3 types de points (core, border, noise)
- [ ] Je comprends paramètres ε et MinPts
- [ ] Je sais utiliser k-distance plot pour ε
- [ ] Je connais avantages (formes arbitraires, détecte bruit)
- [ ] Je sais limitation (densités uniformes)

### ✅ Comparaisons
- [ ] Je sais différence supervisé vs non supervisé
- [ ] Je comprends K-means vs DBSCAN
- [ ] Je sais quand utiliser chaque algorithme
- [ ] Je comprends overfitting vs underfitting
- [ ] Je connais trade-off biais-variance

---

**Bon courage pour votre examen ! 🎓**

*Document théorique basé sur vos besoins et questions d'examen*
