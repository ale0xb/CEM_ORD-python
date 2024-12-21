from cem_ord.cem_ord import CEMOrd
import pandas as pd 


# Load gold data from tsv file it has 3 fields (source, case, label)

gold = pd.read_csv('/workspaces/playground/CEM_ORD-python/data/GOLD.tsv', sep='\t')
# Rename the columns to "source", "case", "label"
gold.columns = ["source", "case", "label"]
sys = pd.read_csv('/workspaces/playground/CEM_ORD-python/data/SYS.tsv', sep='\t')
sys.columns = ["source", "case", "pred"]

# Remove any rows with missing values in pred 
sys = sys.dropna(subset=["pred"])

# Convert 'case' columns to string type to ensure they match
gold["case"] = gold["case"].astype("Int64")
sys["case"] = sys["case"].astype("Int64")

# Merge the gold and sys dataframes on "source" and "case" 
df = pd.merge(gold, sys, on=["source", "case"])

# Finally, convert the labels to string
df["label"] = df["label"].astype(str)
df["pred"] = df["pred"].astype(str)

scores = [] 
# For each source, compute the CEM-Ord score
cem_ord_scores = []
for source in df["source"].unique():
    source_df = df[df["source"] == source]
    gold_labels = source_df["label"].tolist()
    system_labels = source_df["pred"].tolist()
    cem_ord_metric = CEMOrd(gold_labels, system_labels)
    score = cem_ord_metric.evaluate()
    scores.append((source, score))
    print(f"Source: {source}, CEM-Ord Score: {score}")

# Print mean CEM-Ord score
mean_cem_ord_score = sum(score for _, score in scores) / len(scores)
print(f"Mean CEM-Ord Score: {mean_cem_ord_score}")

