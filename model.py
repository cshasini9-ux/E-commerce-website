"""
model.py
--------
AI Recommendation Engine for the Smart E-Commerce Product Recommender.

This module contains all the machine-learning / data-processing logic used
by the Streamlit app (app.py). Keeping this logic separate from the UI code
makes the project easier to read, test and extend.

Core ideas used here:
1. Preference-Based Recommendation (K-Nearest Neighbors)
   - Every product is converted into a small numeric feature vector:
     [normalized_price, rating, normalized_popularity, category_encoded]
   - The user's preferences (category, budget, desired rating) are converted
     into the same kind of feature vector.
   - We fit a NearestNeighbors model on the product feature vectors and find
     the products whose vectors are "closest" to the user's preference
     vector. Distance is converted into a 0-100% "match score" so it is easy
     to understand in the UI.

2. Content-Based Similar Products (Cosine Similarity)
   - When viewing a single product, we recommend other products that are
     similar to it using cosine similarity over the same feature space plus
     the product category (one-hot encoded).

Both techniques are intentionally lightweight (scikit-learn's NearestNeighbors
and cosine_similarity) so the app stays fast, dependency-light, and easy to
understand for an intermediate-level project.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ProductRecommender:
    """
    Wraps a product catalog (pandas DataFrame) and exposes methods for:
    - Preference-based recommendations (KNN)
    - Similar-product recommendations (cosine similarity)
    - Basic analytics helpers used by the dashboard
    """

    def __init__(self, products_df: pd.DataFrame):
        # Keep a clean copy of the original data (used for display)
        self.products = products_df.reset_index(drop=True).copy()

        # Build the numeric feature matrix used by the ML models
        self._build_feature_matrix()

        # Fit a KNN model once so every recommendation call is fast
        self.knn_model = NearestNeighbors(
            n_neighbors=min(10, len(self.products)), metric="euclidean"
        )
        self.knn_model.fit(self.feature_matrix)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_feature_matrix(self):
        """
        Converts the raw product table into a numeric matrix suitable for
        scikit-learn models.

        Features used:
            - Price (scaled 0-1)
            - Rating (already 0-5, scaled 0-1)
            - PopularityScore (scaled 0-1)
            - Category (one-hot encoded, then weighted) -- one-hot encoding
              (rather than a single label-encoded column) matters here: a
              single numeric "category code" would make cosine similarity
              treat every product as pointing in nearly the same direction
              (since all features are positive), making unrelated categories
              look artificially similar. One-hot columns give category real
              separating power in both the KNN and cosine-similarity models.
        """
        df = self.products.copy()

        self.categories = sorted(df["Category"].unique().tolist())
        self.category_to_code = {cat: i for i, cat in enumerate(self.categories)}

        # Scale numeric columns to a common 0-1 range so no single feature
        # (like Price, which has a much larger raw range) dominates the
        # distance calculation.
        self.scaler = MinMaxScaler()
        numeric_cols = ["Price", "Rating", "PopularityScore"]
        scaled_numeric = self.scaler.fit_transform(df[numeric_cols])

        # One-hot encode category, then weight that block more heavily than
        # any single numeric feature so "what kind of product is this"
        # meaningfully drives both nearest-neighbor matching and similarity
        # ranking -- matching shopper intuition (someone browsing
        # Electronics rarely wants a Grocery item, even if price lines up).
        category_dummies = pd.get_dummies(df["Category"])
        category_dummies = category_dummies.reindex(columns=self.categories, fill_value=0)
        category_weight = 0.8
        category_encoded = category_dummies.values.astype(float) * category_weight

        self.feature_matrix = np.hstack([scaled_numeric, category_encoded])
        self.feature_columns = numeric_cols  # numeric cols only; used when building query vectors
        self.category_weight = category_weight

    # ------------------------------------------------------------------
    # Preference-based recommendations (KNN)
    # ------------------------------------------------------------------
    def recommend_from_preferences(
        self,
        preferred_category: str,
        min_budget: float,
        max_budget: float,
        preferred_rating: float,
        top_n: int = 6,
    ) -> pd.DataFrame:
        """
        Recommends products using a K-Nearest Neighbors search.

        The user's stated preferences are converted into a synthetic
        "ideal product" feature vector, then we find the real products in
        the catalog that are numerically closest to that ideal vector.

        Returns a DataFrame of the top_n recommended products with an
        added 'MatchScore' column (0-100%).
        """
        # Represent the "ideal product" using the midpoint of the budget
        # range and the user's desired rating / popularity expectation.
        target_price = (min_budget + max_budget) / 2
        target_rating = preferred_rating
        target_popularity = self.products["PopularityScore"].mean()

        ideal_numeric = pd.DataFrame(
            [[target_price, target_rating, target_popularity]],
            columns=self.feature_columns,
        )
        ideal_numeric_scaled = self.scaler.transform(ideal_numeric)

        # Build the one-hot category block for the ideal vector: the
        # preferred category gets the full category weight, every other
        # category gets 0 -- exactly like a real product's encoding.
        ideal_category = np.zeros((1, len(self.categories)))
        if preferred_category in self.categories:
            cat_index = self.categories.index(preferred_category)
            ideal_category[0, cat_index] = self.category_weight

        ideal_vector = np.hstack([ideal_numeric_scaled, ideal_category])

        # Ask KNN for the closest products to this ideal vector
        n_neighbors = min(top_n * 3, len(self.products))  # over-fetch, then filter/rank
        distances, indices = self.knn_model.kneighbors(
            ideal_vector, n_neighbors=n_neighbors
        )

        results = self.products.iloc[indices[0]].copy()
        results["Distance"] = distances[0]

        # Convert distance into an intuitive 0-100% match score.
        # Smaller distance = higher match. We normalize using the max
        # observed distance in this result set to keep the scale relative.
        max_dist = results["Distance"].max() if results["Distance"].max() > 0 else 1
        results["MatchScore"] = (1 - (results["Distance"] / max_dist)) * 100
        results["MatchScore"] = results["MatchScore"].round(1)

        # Soft-boost products that actually match the requested category
        # and fall within the requested budget, so the ranking "feels"
        # aligned with what the user asked for, not just raw math.
        results["CategoryMatch"] = results["Category"] == preferred_category
        results["BudgetMatch"] = results["Price"].between(min_budget, max_budget)
        results["BoostedScore"] = results["MatchScore"] \
            + results["CategoryMatch"].astype(int) * 8 \
            + results["BudgetMatch"].astype(int) * 5
        results["BoostedScore"] = results["BoostedScore"].clip(upper=99.9).round(1)

        results = results.sort_values("BoostedScore", ascending=False).head(top_n)
        results = results.rename(columns={"BoostedScore": "MatchScore_Final"})

        return results.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Content-based "similar products" (cosine similarity)
    # ------------------------------------------------------------------
    def similar_products(self, product_id: int, top_n: int = 4) -> pd.DataFrame:
        """
        Given a ProductID, returns the top_n most similar products using
        cosine similarity over the numeric feature matrix.
        """
        idx_series = self.products.index[self.products["ProductID"] == product_id]
        if len(idx_series) == 0:
            return pd.DataFrame(columns=self.products.columns)

        idx = idx_series[0]
        sims = cosine_similarity(
            self.feature_matrix[idx].reshape(1, -1), self.feature_matrix
        )[0]

        similarity_df = self.products.copy()
        similarity_df["Similarity"] = sims * 100
        similarity_df["Similarity"] = similarity_df["Similarity"].round(1)

        # Exclude the product itself, then take the top N most similar
        similarity_df = similarity_df[similarity_df["ProductID"] != product_id]
        similarity_df = similarity_df.sort_values("Similarity", ascending=False).head(top_n)

        return similarity_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------
    def category_distribution(self) -> pd.DataFrame:
        """Returns product counts per category (for a pie/bar chart)."""
        return (
            self.products.groupby("Category")
            .size()
            .reset_index(name="ProductCount")
            .sort_values("ProductCount", ascending=False)
        )

    def price_stats(self) -> pd.DataFrame:
        """Returns average price per category (for a bar chart)."""
        return (
            self.products.groupby("Category")["Price"]
            .mean()
            .round(2)
            .reset_index(name="AveragePrice")
            .sort_values("AveragePrice", ascending=False)
        )

    def rating_distribution(self) -> pd.Series:
        """Returns the raw Rating column (for a histogram)."""
        return self.products["Rating"]
