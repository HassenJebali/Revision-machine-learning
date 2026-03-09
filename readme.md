# 📚 Guide de Révision - Machine Learning

**Examen dans 12 heures** - Résumé des concepts clés basé sur vos TPs

---

## 📋 Table des Matières
1. [CRISP-DM](#1-crisp-dm)
2. [Data Preprocessing](#2-data-preprocessing)
3. [ACP (PCA)](#3-acp-pca)
4. [K-Nearest Neighbors (KNN)](#4-k-nearest-neighbors-knn)
5. [Support Vector Machine (SVM)](#5-support-vector-machine-svm)
6. [K-means](#6-k-means)
7. [DBSCAN](#7-dbscan)
8. [Comparaison des Algorithmes](#8-comparaison-des-algorithmes)

---

## 1. CRISP-DM

**C**ross **I**ndustry **S**tandard **P**rocess for **D**ata **M**ining

### 📌 Concept Fondamental

CRISP-DM est une **méthodologie standard** pour mener des projets de Data Mining et Machine Learning.

**⚠️ QUESTION TYPE EXAMEN** : 
> *"Quelle affirmation décrit le mieux la logique générale de la méthodologie CRISP-DM ?"*
> 
> **RÉPONSE : B** - Un cycle itératif favorisant les retours entre les différentes phases du projet.

**Pourquoi ?** CRISP-DM n'est PAS linéaire ! On peut revenir à une phase précédente si nécessaire.

### 📌 Les 6 Phases (ORDRE IMPORTANT) :

#### 1. **Business Understanding** 🎯
**Objectif** : Comprendre ce que l'entreprise veut accomplir

**Questions à se poser :**
- Quel est le problème métier ?
- Quels sont les objectifs ?
- Quels critères de succès ?

**Exemple concret** :
- *Projet* : Détecter les intrusions réseau (KDD'99)
- *Objectif métier* : Protéger le réseau contre les attaques
- *Question ML* : Comment classifier automatiquement les connexions (normales vs attaques) ?

#### 2. **Data Understanding** 📊
**Objectif** : Explorer et comprendre les données disponibles

**Actions clés :**
```python
df.shape              # Nombre de lignes/colonnes
df.info()             # Types de données, mémoire
df.describe()         # Statistiques descriptives
df.head()             # Premières lignes
df.isnull().sum()     # Valeurs manquantes
df['colonne'].value_counts()  # Distribution
```

**Exemple** :
- Dataset KDD'99 : ~150,000 connexions réseau
- 42 features (durée, protocole, nombre d'octets...)
- Variable cible : normal / attaque (23 types d'attaques)

#### 3. **Data Preparation** 🧹
**Objectif** : Préparer les données pour la modélisation

**Activités principales :**
- Nettoyage (NA, duplicatas, outliers)
- Transformation (encodage, normalisation)
- Feature engineering (créer nouvelles variables)
- Sélection de features

**C'est la phase la plus longue** : 60-80% du temps !

#### 4. **Modeling** 🤖
**Objectif** : Construire le modèle ML

**Actions :**
- Choisir l'algorithme adapté (KNN, SVM, K-means...)
- Entraîner le modèle
- Tester différents paramètres

**Exemple** :
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

#### 5. **Evaluation** ✅
**Objectif** : Mesurer la performance du modèle

**Métriques selon le type de problème :**
- Classification : Accuracy, Precision, Recall, F1-Score
- Clustering : Silhouette Score, Inertie

**Retour possible** : Si résultats insuffisants → retour Data Preparation ou Modeling

#### 6. **Deployment** 🚀
**Objectif** : Mettre le modèle en production

**Exemples :**
- API pour détecter spam en temps réel
- Dashboard de segmentation clients
- Système d'alerte intrusion réseau

### 🔄 Caractère Itératif (CRUCIAL POUR L'EXAMEN)

```
Business → Data Understanding → Data Preparation
    ↑            ↓                      ↓
    ↑         Modeling  ←──────  Evaluation
    ↑            ↓
    └──── Deployment
```

**Points clés à retenir** :
- ✅ **Cycles et retours** entre phases
- ✅ **Non linéaire** : on peut revenir en arrière
- ✅ Phase 3 (Data Preparation) = la plus longue
- ✅ Évaluation peut ramener à Preparation ou Modeling
- ❌ Ce n'est PAS strictement séquentiel
- ❌ Ce n'est PAS uniquement technique (phase Business importante)

### 💡 Exemple Complet de Cycle Itératif

**Itération 1** :
1. Business : Détecter fraudes bancaires
2. Data : Explorer transactions (montants, dates, lieux...)
3. Preparation : Normaliser, encoder
4. Modeling : KNN avec K=5
5. Evaluation : Accuracy = 75% ❌ (insuffisant !)

**→ Retour Itération 2** :
3. Preparation : Ajouter feature engineering (heure de transaction, jour semaine)
4. Modeling : SVM avec kernel RBF
5. Evaluation : Accuracy = 92% ✅ (meilleur !)

6. Deployment : API en production

---

## 🤖 Machine Learning vs Programmation Traditionnelle

### **⚠️ QUESTION TYPE EXAMEN**

> *"Quelle est la différence fondamentale entre la programmation traditionnelle et le Machine Learning ?"*
> 
> **RÉPONSE : C** - En Machine Learning, le système apprend les règles à partir des données plutôt que de les coder explicitement.

### 📊 Comparaison Visuelle

**Programmation Traditionnelle** :
```
Données + Règles (codées) → Programme → Résultats
```
*Exemple* : 
```python
if montant > 10000:
    return "fraude"  # Règle codée explicitement
```

**Machine Learning** :
```
Données + Résultats (labels) → Algorithme ML → Règles (apprises)
```
*Exemple* :
```python
modele.fit(transactions, labels_fraude)  # Apprend les règles
# Le modèle découvre lui-même que montant > 10000 ET heure=3h → fraude
```

### 🎯 Points Clés

| Critère | Programmation Classique | Machine Learning |
|---------|------------------------|------------------|
| **Règles** | Définies manuellement | Apprises automatiquement |
| **Adaptation** | Modifier code manuellement | S'adapte aux nouvelles données |
| **Complexité** | Difficile pour problèmes complexes | Gère bien la complexité |
| **Exemple** | Calculatrice, CRUD | Détection spam, reconnaissance faciale |

---

## 📚 Types d'Apprentissage

### **⚠️ QUESTION TYPE EXAMEN**

> *"Vous avez deux projets : regrouper clients selon comportement (sans labels) et détecter spam (avec labels). Quelle différence ?"*
> 
> **RÉPONSE : B** - Le regroupement se fait sans étiquettes (non supervisé), la détection spam avec étiquettes (supervisé).

### 1️⃣ Apprentissage Supervisé

**Définition** : On a les **réponses** (labels) dans les données d'entraînement

**Exemples** :
- ✉️ Email spam ou non ? → Classification binaire
- 🏠 Prix d'une maison ? → Régression
- 🔒 Détection d'intrusions réseau → Classification multi-classe

**Algorithmes** : KNN, SVM, Régression Linéaire, Decision Trees

**Données** :
```python
X = features  # Variables explicatives
y = labels    # Variable cible (CONNUE)
```

### 2️⃣ Apprentissage Non Supervisé

**Définition** : **Pas de labels** ! On cherche des structures cachées

**Exemples** :
- 👥 Segmentation clients → Clustering
- 📊 Réduction dimensionnalité → PCA
- 🔍 Détection d'anomalies

**Algorithmes** : K-means, DBSCAN, PCA, Hierarchical Clustering

**Données** :
```python
X = features  # Variables explicatives
# PAS de y ! Le modèle trouve les groupes tout seul
```

### 3️⃣ Apprentissage par Renforcement

**⚠️ QUESTION TYPE EXAMEN** :
> *"Robot qui apprend à maximiser récompenses après chaque action réussie ?"*
> 
> **RÉPONSE : C** - Apprentissage par renforcement

**Définition** : L'agent apprend par **essais-erreurs** avec système de récompenses

**Composants** :
- **Agent** : Le système qui apprend (robot, IA jeu vidéo)
- **Environnement** : Le monde où l'agent évolue
- **Actions** : Ce que l'agent peut faire
- **Récompenses** : +points (bonnes actions) ou -points (mauvaises)

**Exemples** :
- 🤖 Robot qui apprend à marcher
- 🎮 IA qui joue aux échecs (AlphaGo)
- 🚗 Voiture autonome

**Différence clé** :
- Supervisé : "Voici la bonne réponse"
- Renforcement : "Tu as fait bien/mal, continue d'essayer"

### 4️⃣ Apprentissage Semi-Supervisé

**Un mix** : Peu de données labellisées + beaucoup de données non labellisées

**Exemple** : 100 images de chats étiquetées + 10,000 images non étiquetées

---

## ⚖️ Overfitting vs Underfitting

### **⚠️ QUESTIONS TYPE EXAMEN**

> *"Quelle stratégie réduit le sur-apprentissage (overfitting) ?"*
> 
> **RÉPONSE : C** - Augmenter la taille du jeu d'entraînement

> *"Parmi les courbes, laquelle illustre underfitting ?"*
> 
> **RÉPONSE** : La courbe trop simple qui ne capture pas la tendance des données

### 📊 Visualisation

**Underfitting** (Sous-apprentissage) :
```
Données: Points en forme de courbe
Modèle: Ligne droite qui passe loin des points
→ Trop simple, erreur élevée sur train ET test
```
**Caractéristiques** :
- ❌ Accuracy train faible
- ❌ Accuracy test faible  
- ❌ Modèle trop simple
- 📉 **High Bias**, Low Variance

**Good Fit** (Ajustement optimal) :
```
Données: Points en forme de courbe
Modèle: Courbe qui suit bien les points
→ Capture la tendance, erreur raisonnable
```
**Caractéristiques** :
- ✅ Accuracy train élevée
- ✅ Accuracy test élevée
- ✅ Généralise bien
- ⚖️ Équilibre Bias-Variance

**Overfitting** (Sur-apprentissage) :
```
Données: Points en forme de courbe avec bruit
Modèle: Courbe complexe passant par TOUS les points
→ Mémorise le bruit, ne généralise pas
```
**Caractéristiques** :
- ✅ Accuracy train très élevée (99%)
- ❌ Accuracy test faible (60%)
- ❌ Trop complexe, mémorise le bruit
- 📉 Low Bias, **High Variance**

### 🛠️ Solutions aux Problèmes

**Pour Underfitting** (modèle trop faible) :
- ✅ Augmenter complexité du modèle
- ✅ Ajouter plus de features
- ✅ Réduire régularisation
- ✅ Entraîner plus longtemps

**Pour Overfitting** (modèle trop fort) :
- ✅ **Augmenter taille données d'entraînement** ⭐
- ✅ Réduire complexité du modèle
- ✅ Utiliser régularisation (L1, L2)
- ✅ Early stopping
- ✅ Cross-validation
- ❌ Ne PAS augmenter complexité
- ❌ Ne PAS réduire données d'entraînement

### 💡 Exemple Concret

**Problème** : Prédire notes élèves selon heures travail

**Underfitting** :
```python
# Modèle trop simple : moyenne constante
prediction = 12  # pour tous les élèves
# Erreur train = 3.5, Erreur test = 3.4
```

**Good Fit** :
```python
# Régression linéaire simple
note = 2 * heures_travail + 5
# Erreur train = 0.8, Erreur test = 0.9
```

**Overfitting** :
```python
# Polynôme degré 20 qui passe par TOUS les points
note = 0.001*x^20 - 0.5*x^19 + ...
# Erreur train = 0.01, Erreur test = 5.2 ❌
```

---

## 2. Data Preprocessing

### 🔍 Étapes Essentielles

#### A. Exploration Initiale
```python
df.shape              # Dimensions
df.info()             # Types de données
df.describe()         # Statistiques descriptives
df.isnull().sum()     # Valeurs manquantes
df.duplicated().sum() # Duplicatas
```

#### B. Nettoyage des Données

**1. Valeurs manquantes**
```python
df.dropna()                    # Supprimer lignes avec NA
df.fillna(value)               # Remplir avec une valeur
```

**2. Duplicatas**
```python
df.drop_duplicates(inplace=True)
```

**3. Valeurs aberrantes**
- Utiliser `.describe()` pour identifier
- Visualiser avec histogrammes

#### C. Gestion des Corrélations
```python
corr = df.corr(numeric_only=True)
```
- **Corrélation proche de 1** : Variables redondantes → Supprimer une
- **Corrélation proche de -1** : Corrélation négative forte
- **Corrélation proche de 0** : Pas de relation linéaire

> **❗ Dans vos TPs** : Suppression de colonnes avec corrélation > 0.97

#### D. Encodage des Variables Catégorielles

**1. Label Encoding** (pour la variable cible)
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])
```

**2. Mapping Manuel** (pour peu de catégories)
```python
protocol_map = {'icmp': 0, 'tcp': 1, 'udp': 2}
df['protocol'] = df['protocol'].map(protocol_map)
```

**3. One-Hot Encoding** (pour plusieurs catégories)
```python
from sklearn.preprocessing import OneHotEncoder
```

#### E. Normalisation/Standardisation

**StandardScaler** ⭐ (LE PLUS IMPORTANT)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Pourquoi ?**
- Algorithmes sensibles à l'échelle : KNN, SVM, K-means, DBSCAN
- Met toutes les features sur la même échelle
- Moyenne = 0, Écart-type = 1

> **🎯 Règle d'or** : Toujours normaliser avant KNN, SVM, K-means, DBSCAN

#### F. Séparation Train/Test
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 3. ACP (PCA)

**Analyse en Composantes Principales** = Principal Component Analysis

### 📌 Concept
- Technique de **réduction de dimensionnalité**
- Transforme les features corrélées en composantes principales non corrélées
- Conserve le maximum de variance

### 🎯 Objectifs
1. **Réduire** le nombre de features
2. **Visualiser** des données multidimensionnelles
3. **Accélérer** les algorithmes
4. **Éviter** le surapprentissage

### 💻 Utilisation
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Garder 2 composantes
X_pca = pca.fit_transform(X_scaled)

# Variance expliquée
print(pca.explained_variance_ratio_)
```

### ⚠️ Points Importants
- **Toujours standardiser** avant PCA
- Choisir le nombre de composantes selon la variance expliquée cumulée (ex: 95%)
- PCA perd l'interprétabilité des features

---

## 4. K-Nearest Neighbors (KNN)

### 📌 Concept
- Algorithme **supervisé** de classification
- Prédit la classe en fonction des **K voisins les plus proches**
- Basé sur la **distance** (euclidienne généralement)

### 🎯 Principe
1. Pour un nouveau point, trouver les K voisins les plus proches
2. Voter : la classe majoritaire parmi les K voisins gagne

### 💻 Implémentation
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### ⚙️ Paramètres Clés
- **n_neighbors (K)** : Nombre de voisins
  - K trop petit → Sensible au bruit, overfitting
  - K trop grand → Underfitting
  - **Méthode** : Tester plusieurs valeurs (boucle)

### 📊 Trouver le Meilleur K
```python
for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
```

### ✅ Évaluation
```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
```

### ⚠️ Points Importants
- **Sensible à l'échelle** → TOUJOURS standardiser
- **Lent sur grandes données** (calcul de distances)
- Bon pour données avec frontières claires

---

## 5. Support Vector Machine (SVM)

### 📌 Concept
- Algorithme **supervisé** de classification
- Trouve l'**hyperplan optimal** qui sépare les classes
- Maximise la **marge** entre les classes

### 🎯 Principe
- Chercher la frontière de décision qui maximise la distance aux points les plus proches (support vectors)

### 💻 Implémentation
```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf')  # kernel RBF (Radial Basis Function)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
```

### ⚙️ Paramètres Clés

**1. Kernel** (noyau)
- **'linear'** : Séparation linéaire (lignes droites)
- **'rbf'** : ⭐ Le plus utilisé, pour données non linéaires
- **'poly'** : Polynomial
- **'sigmoid'**

**2. C (régularisation)**
- C petit → Marge large, peut mal classer certains points
- C grand → Marge étroite, risque d'overfitting

**3. gamma (pour RBF)**
- Gamma petit → Influence large
- Gamma grand → Influence locale

### ✅ Évaluation
```python
accuracy = svm.score(X_test, y_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### ⚠️ Points Importants
- **Sensible à l'échelle** → Standardiser obligatoire
- Performant sur données de haute dimension
- Bon pour séparations complexes (avec kernel rbf)
- Plus lent que KNN sur grandes données

---

## 6. K-means

### 📌 Concept
- Algorithme de **clustering NON supervisé**
- Regroupe les données en **K clusters** selon leur similarité
- Basé sur la **distance** aux centres (centroids)

### 🎯 Principe (Algorithme)
1. Choisir K centres initiaux (aléatoire ou k-means++)
2. Assigner chaque point au centre le plus proche
3. Recalculer les centres (moyenne des points du cluster)
4. Répéter 2-3 jusqu'à convergence

### 💻 Implémentation
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=12, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
```

### ⚙️ Paramètres Clés

**1. n_clusters** : Nombre de clusters
- **Comment choisir ?**
  - Méthode du coude (Elbow Method)
  - Silhouette Score
  
**2. init** : Initialisation
- **'k-means++'** : ⭐ Recommandé, initialisation intelligente
- **'random'** : Aléatoire

**3. n_init** : Nombre d'exécutions
- Le meilleur résultat est conservé

### 📊 Trouver le Nombre Optimal de Clusters

**Méthode 1 : Silhouette Score** ⭐
```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
```
- Score entre -1 et 1
- **Plus proche de 1** = meilleur

**Méthode 2 : Inertia (somme des distances²)**
```python
inertia = kmeans.inertia_
```

### ✅ Visualisation
```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.show()
```

### ⚠️ Points Importants
- **Sensible aux valeurs aberrantes**
- **Fonctionne mal** avec clusters de formes non sphériques
- **Sensible à l'initialisation** → Utiliser k-means++
- **Standardiser** les données avant

---

## 7. DBSCAN

**Density-Based Spatial Clustering of Applications with Noise**

### 📌 Concept
- Algorithme de **clustering basé sur la densité**
- Trouve des clusters de **formes arbitraires**
- Identifie automatiquement les **points de bruit** (outliers)

### 🎯 Principe
- Un point est dans un cluster si :
  1. Il a au moins **min_samples** voisins dans un rayon **epsilon**
  2. Ou il est voisin d'un tel point

### 💻 Implémentation
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.15, min_samples=5)
dbscan.fit(X)
labels = dbscan.labels_

# -1 = bruit/outliers
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
```

### ⚙️ Paramètres Clés

**1. eps (epsilon)** : Rayon de voisinage
- **Comment choisir ?** Méthode du graphe k-distance
```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X)
distances, _ = neighbors.kneighbors(X)
distances = np.sort(distances[:, -1])

# Tracer le graphe et chercher le "coude"
plt.plot(distances)
plt.show()
```

**2. min_samples** : Nombre minimum de points
- Généralement : 5 ou 2 × dimensions

### 📊 Déterminer Epsilon (IMPORTANT)
1. Calculer distance au k-ème voisin pour chaque point
2. Trier les distances
3. Tracer le graphe (k-distance plot)
4. Chercher le **point de coude** (changement de pente)

### ✅ Avantages vs K-means
- ✅ Trouve clusters de **formes non convexes**
- ✅ **Détecte le bruit** automatiquement
- ✅ **Pas besoin** de spécifier le nombre de clusters
- ✅ Résistant aux outliers

### ❌ Inconvénients
- ❌ Sensible aux paramètres (eps, min_samples)
- ❌ Difficile avec densités variables
- ❌ Plus lent sur grandes données

### 🆚 Quand utiliser DBSCAN ?
- Formes de clusters complexes (demi-lunes, spirales...)
- Nombre de clusters inconnu
- Présence de bruit/outliers
- Densités relativement uniformes

---

## 8. Comparaison des Algorithmes

### 📊 Tableau Comparatif

| Critère | KNN | SVM | K-means | DBSCAN |
|---------|-----|-----|---------|---------|
| **Type** | Supervisé | Supervisé | Non supervisé | Non supervisé |
| **Usage** | Classification | Classification | Clustering | Clustering |
| **Paramètre clé** | K (voisins) | Kernel, C | n_clusters | eps, min_samples |
| **Standardisation** | ✅ Obligatoire | ✅ Obligatoire | ✅ Recommandé | ✅ Recommandé |
| **Formes complexes** | ❌ | ✅ (avec rbf) | ❌ (sphériques) | ✅ |
| **Détection bruit** | ❌ | ❌ | ❌ | ✅ |
| **Nombre clusters** | N/A | N/A | À spécifier | Automatique |
| **Vitesse** | Lent | Moyen | Rapide | Moyen |
| **Grandes données** | ❌ | ❌ | ✅ | ❌ |

### 🎯 Choix de l'Algorithme

**Classification (supervisé - avec labels)**
- **KNN** : Données simples, frontières claires, peu de features
- **SVM** : Données complexes, haute dimension, séparation non linéaire

**Clustering (non supervisé - sans labels)**
- **K-means** : Nombre de clusters connu, clusters sphériques, rapidité
- **DBSCAN** : Formes complexes, détection outliers, nombre inconnu

---

## 🔑 Points Clés à Retenir

### ⚠️ Avant Tout Algorithme
1. **Explorer** les données (shape, info, describe)
2. **Nettoyer** (NA, duplicatas)
3. **Encoder** les variables catégorielles
4. **Standardiser** avec StandardScaler
5. **Séparer** train/test (pour supervisé)

### 📌 Distinctions Importantes

**Supervisé vs Non Supervisé**
- **Supervisé** (KNN, SVM) : Données avec labels → Prédiction
- **Non supervisé** (K-means, DBSCAN) : Données sans labels → Regroupement

**Distance vs Densité**
- **Distance** (KNN, K-means) : Basés sur distances euclidiennes
- **Densité** (DBSCAN) : Basé sur concentration de points

**Paramètres Critiques**
- **KNN** : K (nombre de voisins)
- **SVM** : kernel (type de séparation)
- **K-means** : n_clusters (nombre de groupes)
- **DBSCAN** : eps et min_samples (rayon et densité)

### 🎓 Métriques d'Évaluation

**Supervisé**
```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

**Non Supervisé**
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)  # Entre -1 et 1, 1 = meilleur
```

---

## 💡 Astuces pour l'Examen

### ✅ Questions Fréquentes (QCM)

1. **Quel algorithme pour clusters non convexes ?**
   → DBSCAN

2. **Quel preprocessing avant KNN/SVM ?**
   → StandardScaler (normalisation)

3. **Comment trouver le meilleur K pour KNN ?**
   → Tester plusieurs valeurs et comparer accuracy

4. **Différence K-means vs DBSCAN ?**
   → K-means: clusters sphériques, nombre fixé
   → DBSCAN: formes complexes, détecte bruit

5. **Kernel SVM pour données non linéaires ?**
   → rbf

6. **Métrique pour évaluer K-means ?**
   → Silhouette Score, Inertia

7. **Label -1 dans DBSCAN signifie ?**
   → Point de bruit (outlier)

8. **Phases CRISP-DM dans l'ordre ?**
   → Business → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment

9. **Rôle de StandardScaler ?**
   → Mettre features sur même échelle (moyenne=0, std=1)

10. **KNN est supervisé ou non ?**
    → Supervisé

### 🚀 Checklist Révision Express

- [ ] Je connais les 6 phases de CRISP-DM
- [ ] Je sais quand utiliser StandardScaler
- [ ] Je peux expliquer KNN (distance, voisins)
- [ ] Je connais les kernels SVM (linear, rbf)
- [ ] Je sais la différence K-means / DBSCAN
- [ ] Je peux calculer le meilleur K (KNN ou K-means)
- [ ] Je sais trouver epsilon pour DBSCAN
- [ ] Je connais les métriques (accuracy, silhouette_score)
- [ ] Je sais gérer les variables catégorielles
- [ ] Je peux identifier supervisé vs non supervisé

---

## 📝 Code Templates Rapides

### Template KNN
```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate
accuracy = knn.score(X_test, y_test)
```

### Template SVM
```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = svm.score(X_test, y_test)
```

### Template K-means
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Trouver meilleur k
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"k={k}, silhouette={score}")
```

### Template DBSCAN
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
```

---

## 🎯 Bon Courage pour l'Examen !

**Rappels finaux :**
1. Lisez bien les questions (QCM = une seule réponse)
2. Pensez au preprocessing (StandardScaler)
3. Supervisé ≠ Non supervisé
4. K-means ≠ DBSCAN (formes, bruit)
5. Restez calme et confiant ! 💪

---

*Document créé à partir de vos notebooks de TP*  
*Bonne chance ! 🍀*
