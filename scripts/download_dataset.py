
from sklearn.datasets import fetch_openml

print("Downloading the real credit-card dataset...")
print("The download may take a few minutes.")

dataset = fetch_openml(
    "creditcard",
    version=1,
    as_frame=True,
)

df = dataset.frame
df["Class"] = df["Class"].astype(int)

df.to_csv("data/creditcard.csv", index=False)

print("\nDownload complete!")
print("Dataset shape:", df.shape)
print("\nFraud labels:")
print(df["Class"].value_counts())
print("\nSaved to data/creditcard.csv")
