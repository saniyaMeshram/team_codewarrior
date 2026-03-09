import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Reduce dataset size (keep only first 20000 rows from each file)
fake = fake.head(2000)
true = true.head(2000)

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data = data[["title","text","label"]]

data.to_csv("news.csv", index=False)

print("Dataset combined successfully and reduced!")