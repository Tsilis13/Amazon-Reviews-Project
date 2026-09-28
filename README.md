# Amazon Reviews Analysis

Text mining and machine learning on the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset (McAuley Lab). The project covers exploratory analysis, sentiment feature engineering, product clustering, recommender systems and review sentiment classification across five product categories: **Automotive, Baby Products, Musical Instruments, Home & Kitchen, Sports & Outdoors**.

## Contents

| Notebook | Description |
|---|---|
| [`Amazon_Reviews_Analysis_Part_1.ipynb`](Amazon_Reviews_Analysis_Part_1.ipynb) | Data preparation, exploratory analysis, sentiment scoring, weighted product ratings |
| [`Amazon_Reviews_Analysis_Part_2.ipynb`](Amazon_Reviews_Analysis_Part_2.ipynb) | Product clustering, recommender systems, sentiment classification |

## Data

Reviews and product metadata are streamed from Hugging Face and cleaned once, then cached as CSV files.

| Stage | Scope | Size |
|---|---|---|
| Part 1: reviews | 5 categories | 50,000 reviews per category |
| Part 1: metadata | 5 categories | Products with valid price, average rating and description |
| Part 2, Task 1: clustering | 5 categories | 10,000 products per category |
| Part 2, Tasks 2-3: recommender and classifier | Baby Products | 2,818 reviews (after 5-core filtering), 3,000 products |

**Text cleaning (spaCy `en_core_web_sm`):** lower-casing, removal of non-alphabetic characters, stop-word removal and lemmatisation.

## Part 1: Exploration and Feature Engineering

**Exploratory analysis**
- Rating distribution and average rating per category
- Frequent words and bigrams in low-rated products (at least 6 reviews, average rating of 3 or below)
- Key attributes of the top-5 best sellers per category, ranked by number of ratings
- Average yearly rating per category

**Sentiment score.** Each review is scored by combining VADER's compound score with the normalised star rating:

```
final_sentiment_score = 0.35 * VADER_compound + 0.65 * ((rating - 1) / 4)
```

**Weighted product rating.** Average rating is scaled by review volume to reduce the influence of products with few reviews:

```
weighted_rating = average_rating * log(1 + review_count)
```

### Findings

- Ratings are strongly skewed towards 5 stars (about 68-70% of reviews in every category). Average ratings are nearly identical across categories:

| Category | Average rating |
|---|---|
| Automotive | 4.34 |
| Baby Products | 4.38 |
| Musical Instruments | 4.36 |
| Home & Kitchen | 4.37 |
| Sports & Outdoors | 4.38 |

- Yearly averages fluctuate widely before roughly 2011, when review counts were small, and settle between 4.2 and 4.4 from 2013 onwards.
- Low-rated products show category-specific complaints, for example *minor scratch* and *waste money* (Automotive), *car seat* and *night vision* (Baby Products), *sound quality* and *piece junk* (Musical Instruments), *stop work* (Home & Kitchen) and *battery life* (Sports & Outdoors).
- Best sellers are typically low-cost, high-volume everyday items such as car vacuums, wiper blades, diapers and wipes, guitar strings, bed sheets and resistance bands.

## Part 2: Learning Tasks

### Task 1: Product clustering (all categories)

- **Features:** TF-IDF of the cleaned description (max 10,000 terms) combined with min-max scaled price and average rating
- **Model:** K-Means, with K chosen by the elbow method
- **Evaluation:** silhouette score and 2-D PCA projection

| Category | K | Silhouette |
|---|---|---|
| Automotive | 5 | 0.010 |
| Baby Products | 7 | 0.018 |
| Musical Instruments | 5 | 0.008 |
| Home & Kitchen | 8 | 0.012 |
| Sports & Outdoors | 6 | 0.009 |

Silhouette scores near zero indicate overlapping clusters, which is typical for high-dimensional sparse text features.

### Task 2: Recommendation system (Baby Products)

An iterative 5-core filter keeps only users and items with at least five reviews, so the user-item matrix is dense enough for collaborative filtering.

| Method | Approach |
|---|---|
| User-based collaborative filtering | Cosine similarity between users; weighted average of the top-k neighbours' ratings |
| Item-based collaborative filtering | Cosine similarity between items; weighted average over the user's most similar rated items |
| Content-based filtering | Product descriptions embedded with pre-trained GloVe vectors (100-d); cosine similarity between products |
| Hybrid | Weighted average of the three scores (default 0.4 / 0.3 / 0.3), re-normalised when a method returns no score |

Content-based filtering achieves **Recall@15 = 0.0144** (in-sample evaluation).

### Task 3: Sentiment classification (Baby Products)

- **Labels:** positive (4-5 stars), neutral (3 stars), negative (1-2 stars)
- **Features:** TF-IDF vs. averaged GloVe embeddings
- **Models:** Naive Bayes, K-Nearest Neighbours, Random Forest
- **Protocol:** 80/20 train-test split, 10-fold cross-validation on the training set, final evaluation on the held-out test set with macro-averaged metrics

**Test-set results**

| Features | Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| TF-IDF | Naive Bayes | 0.288 | 0.333 | 0.309 | 0.863 |
| TF-IDF | KNN | 0.622 | 0.341 | 0.325 | 0.865 |
| TF-IDF | Random Forest | 0.543 | 0.375 | 0.385 | 0.865 |
| GloVe | Naive Bayes | 0.374 | 0.403 | 0.291 | 0.406 |
| GloVe | KNN | 0.612 | 0.396 | **0.419** | 0.863 |
| GloVe | Random Forest | 0.845 | 0.358 | 0.358 | 0.867 |

The dataset is heavily imbalanced towards positive reviews, so accuracy (about 0.86) is high while macro F1 stays between 0.29 and 0.42; Naive Bayes on TF-IDF, for instance, predicts a single class (macro recall of exactly 1/3). Macro-averaged metrics are therefore the reliable indicator, and KNN on GloVe embeddings performs best on macro F1.

## Limitations and Next Steps

- **Class imbalance:** class weighting or resampling, and transformer-based classifiers.
- **Negation:** VADER is applied to cleaned text where stop words such as "not" have been removed; scoring the raw text would preserve negation.
- **Sampling:** records are the first valid entries of each stream rather than a random sample.
- **Recommender evaluation:** Recall@K is in-sample; a held-out split with ranking metrics (NDCG, MAP) would be more reliable, and matrix factorisation would scale better than dense similarity matrices.
- **Clustering:** dimensionality reduction (LSA) or sentence embeddings before K-Means may improve cluster separation.

## Setup

The notebooks were developed on **Google Colab** and cache intermediate CSVs on Google Drive.

```bash
pip install datasets matplotlib seaborn gensim scikit-learn pandas numpy nltk spacy tabulate scipy
python -m spacy download en_core_web_sm
```

1. Run **Part 1** first to build and cache the five-category CSVs.
2. Set `csv_folder` in the "Load the cached CSVs" cells to your Drive folder.
3. Run **Part 2**. Clustering reuses the Part 1 metadata; the recommender and classifier use the Baby Products CSVs created inside Part 2.

The streaming and spaCy cleaning cells are slow and only need to run once. The GloVe model (`glove-wiki-gigaword-100`) downloads automatically through `gensim.downloader`.

## Tech Stack

Python, pandas, NumPy, SciPy, scikit-learn, spaCy, NLTK (VADER), gensim (GloVe), Hugging Face `datasets`, Matplotlib, seaborn.

## References

- Pennington, J., Socher, R., Manning, C. *GloVe: Global Vectors for Word Representation.* 2014.
- Hutto, C., Gilbert, E. *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* 2014.
