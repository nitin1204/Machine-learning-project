#!/usr/bin/env python3
"""
restaurant_recommender.py

Content-based restaurant recommender with a simple CLI.
- Automatically finds dataset.csv under current working directory.
- Builds TF-IDF profiles from 'Cuisines' + 'City' + 'Price range'.
- Adds numeric signals (Aggregate rating, Votes, Average Cost for two).
- Computes cosine similarity and returns top-N recommendations.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ---------- Config ----------
DATA_FILENAME = "dataset.csv"
TFIDF_MAX_FEATURES = 1000
NUMERIC_FEATURES = ['Aggregate rating', 'Votes', 'Average Cost for two']
ONLINE_BOOST = 0.15   # boost if user prefers online delivery
# ----------------------------

def find_dataset(start_dir=os.getcwd(), filename=DATA_FILENAME):
    """Recursively search for filename starting from start_dir."""
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.copy()
    # Basic cleaning
    df['Cuisines'] = df.get('Cuisines', pd.Series(index=df.index)).fillna('Unknown').astype(str)
    df['City'] = df.get('City', pd.Series(index=df.index)).fillna('Unknown').astype(str)
    # Ensure Price range exists, else create a default
    if 'Price range' not in df.columns:
        df['Price range'] = 0
    df['Price range'] = df['Price range'].fillna(0).astype(int)
    df['Price_range_str'] = df['Price range'].astype(str)
    # Compose textual profile
    df['profile_text'] = df['Cuisines'].str.replace(',', ' ') + ' ' + df['City'] + ' price_' + df['Price_range_str']
    # Numeric features: coerce to numeric and fill missing
    for c in NUMERIC_FEATURES:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    # Normalize Has Online delivery to 0/1
    if 'Has Online delivery' in df.columns:
        online = df['Has Online delivery'].astype(str).str.strip().replace({'Yes':'1','No':'0','yes':'1','no':'0'})
        df['Has Online delivery_numeric'] = pd.to_numeric(online, errors='coerce').fillna(0).astype(int)
    else:
        df['Has Online delivery_numeric'] = 0
    return df

def build_feature_matrices(df):
    # TF-IDF on profile_text
    tfidf = TfidfVectorizer(stop_words='english', max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(df['profile_text'].astype(str))
    # Numeric features scaled to [0,1]
    scaler = MinMaxScaler()
    numeric_array = scaler.fit_transform(df[NUMERIC_FEATURES].values)
    # Combine sparse TF-IDF and dense numeric (converted to sparse horizontally)
    combined = hstack([tfidf_matrix, numeric_array])
    return tfidf, tfidf_matrix, scaler, numeric_array, combined

def build_user_vector(tfidf, scaler, user_cuisines, user_city, user_price, df):
    # Build user text string similar to profile_text
    cuisines_part = ""
    if user_cuisines:
        if isinstance(user_cuisines, str):
            # allow comma-separated string
            cuisines_part = " ".join([c.strip() for c in user_cuisines.split(',') if c.strip()])
        elif isinstance(user_cuisines, (list, tuple)):
            cuisines_part = " ".join([c.strip() for c in user_cuisines])
    city_part = user_city if user_city else ""
    price_part = f"price_{user_price}" if user_price is not None else ""
    user_text = f"{cuisines_part} {city_part} {price_part}".strip()
    user_tfidf = tfidf.transform([user_text])  # shape (1, n_tfidf_features)
    # Numeric baseline: mean of dataset numeric features
    numeric_mean = np.mean(df[NUMERIC_FEATURES].values, axis=0).reshape(1, -1)
    user_numeric = numeric_mean.copy()
    if user_price is not None:
        # attempt to map price to a representative 'Average Cost for two' using medians
        if 'Price range' in df.columns:
            med_costs = df.groupby('Price range')['Average Cost for two'].median().to_dict()
            if user_price in med_costs and not np.isnan(med_costs[user_price]):
                rep_cost = med_costs[user_price]
                # create numeric vector: [avg_rating, median_votes, rep_cost]
                # we keep rating and votes as dataset means/medians
                avg_rating = df['Aggregate rating'].mean() if 'Aggregate rating' in df.columns else 0
                med_votes = df['Votes'].median() if 'Votes' in df.columns else 0
                numeric_candidate = np.array([[avg_rating, med_votes, rep_cost]])
                # scale with scaler (which expects shape (n_samples, n_features))
                try:
                    user_numeric = scaler.transform(numeric_candidate)
                except Exception:
                    # fallback if scaler expects different feature names/shapes -> stick to mean
                    user_numeric = numeric_mean
    # Return combined user vector as dense 1D array shaped to match combined later
    user_combined = np.hstack([user_tfidf.toarray()[0], user_numeric.reshape(-1)])
    return user_combined

def recommend(df, combined_matrix, tfidf, scaler, user_cuisines=None, user_city=None, user_price=None, prefer_online=None, top_n=10):
    user_vec = build_user_vector(tfidf, scaler, user_cuisines, user_city, user_price, df)
    sims = cosine_similarity(combined_matrix, user_vec.reshape(1, -1)).reshape(-1)
    # If prefer_online is specified, boost scores accordingly
    if prefer_online is not None and 'Has Online delivery_numeric' in df.columns:
        online_col = df['Has Online delivery_numeric'].values
        if prefer_online:
            sims = sims + online_col * ONLINE_BOOST
        else:
            sims = sims + (1 - online_col) * (ONLINE_BOOST / 3.0)
    # If city specified, prefer same-city by filtering first; otherwise global
    results_df = df.copy()
    if user_city:
        same_city = results_df[results_df['City'].str.lower() == str(user_city).lower()]
        if not same_city.empty:
            results_df = same_city
            sims = sims[results_df.index.values]  # align sims to filtered df
    # attach scores
    results_df = results_df.assign(score=sims)
    results = results_df.sort_values('score', ascending=False).head(top_n)
    # Friendly output columns
    display_cols = ['Restaurant Name', 'City', 'Cuisines', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes', 'Has Online delivery', 'score']
    cols_present = [c for c in display_cols if c in results.columns]
    return results[cols_present]

def interactive_loop(df, tfidf, scaler, combined_matrix):
    print("\n--- Restaurant Recommender CLI ---")
    print("Type 'exit' at any prompt to quit.\n")
    while True:
        try:
            raw_cuisines = input("Preferred cuisines (comma-separated, e.g. Italian, Chinese) [or blank for none]: ").strip()
            if raw_cuisines.lower() == 'exit': break
            cuisines = raw_cuisines if raw_cuisines else None

            raw_city = input("City (e.g. New Delhi) [or blank for any]: ").strip()
            if raw_city.lower() == 'exit': break
            city = raw_city if raw_city else None

            raw_price = input("Price range (1-4) [or blank for any]: ").strip()
            if raw_price.lower() == 'exit': break
            price = int(raw_price) if raw_price.isdigit() else None

            raw_online = input("Prefer online delivery? (yes/no/blank for don't care): ").strip().lower()
            if raw_online == 'exit': break
            if raw_online == 'yes':
                prefer_online = True
            elif raw_online == 'no':
                prefer_online = False
            else:
                prefer_online = None

            raw_topn = input("How many recommendations? [default 10]: ").strip()
            if raw_topn.lower() == 'exit': break
            topn = int(raw_topn) if raw_topn.isdigit() else 10

            results = recommend(df, combined_matrix, tfidf, scaler,
                                user_cuisines=cuisines, user_city=city,
                                user_price=price, prefer_online=prefer_online,
                                top_n=topn)
            if results.empty:
                print("\nNo matching restaurants found (try different filters).\n")
            else:
                print("\nTop recommendations:\n")
                print(results.to_string(index=False))
                # Optionally save
                save = input("\nSave these results to CSV? (yes/no): ").strip().lower()
                if save == 'yes':
                    outname = f"recommendations_{int(pd.Timestamp.now().timestamp())}.csv"
                    results.to_csv(outname, index=False)
                    print(f"Saved to {outname}")
            cont = input("\nMake another query? (yes/no): ").strip().lower()
            if cont != 'yes':
                break
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}\nTry again or type 'exit' to quit.\n")

def main():
    ds_path = find_dataset()
    if not ds_path:
        print(f"ERROR: {DATA_FILENAME} not found under {os.getcwd()}. Place {DATA_FILENAME} somewhere in this folder tree.")
        sys.exit(1)
    print(f"Found dataset: {ds_path}")
    df = load_and_preprocess(ds_path)
    tfidf, tfidf_matrix, scaler, numeric_array, combined_matrix = build_feature_matrices(df)
    print(f"Dataset loaded: {len(df)} restaurants. TF-IDF shape: {tfidf_matrix.shape}")
    interactive_loop(df, tfidf, scaler, combined_matrix)
    print("Goodbye!")

if __name__ == "__main__":
    main()
