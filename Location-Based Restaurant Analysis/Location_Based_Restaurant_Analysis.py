import pandas as pd
import matplotlib.pyplot as plt

# =============================
# 1. Load dataset
# =============================
df = pd.read_csv("dataset.csv")  # make sure the file is in the same folder

# Basic info
print(df.head())
print(df.columns)

# =============================
# 2. Clean data
# =============================
df = df.dropna(subset=["Latitude", "Longitude"])
df["City"] = df["City"].astype(str).str.strip()
df["Locality"] = df["Locality"].astype(str).str.strip()

# =============================
# 3. Geographical Distribution
# =============================
plt.figure(figsize=(10, 7))
plt.scatter(df["Longitude"], df["Latitude"], alpha=0.4, s=10)
plt.title("Restaurant Geographical Distribution")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# =============================
# 4. Restaurants by City
# =============================
city_counts = df["City"].value_counts()
print(city_counts)

city_counts.head(20).plot(kind="bar", figsize=(12, 6))
plt.title("Number of Restaurants by City")
plt.xlabel("City")
plt.ylabel("Restaurant Count")
plt.show()

# =============================
# 5. Average Rating by City
# =============================
avg_rating_city = df.groupby("City")["Aggregate rating"].mean()
print(avg_rating_city)

avg_rating_city.plot(kind="bar", figsize=(14, 6))
plt.title("Average Restaurant Rating by City")
plt.xlabel("City")
plt.ylabel("Average Rating")
plt.show()
