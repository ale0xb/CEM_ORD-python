from cem_ord.cem_ord import CEMOrd
import pandas as pd 


# Load gold data from tsv file it has 3 fields (source, case, label)

gold = pd.read_csv('data/GOLD.tsv', sep='\t')
# Rename the columns to "source", "case", "label"
gold.columns = ["source", "case", "label"]
sys = pd.read_csv('data/SYS.tsv', sep='\t')
sys.columns = ["source", "case", "pred"]

# Merge the gold and sys dataframes on "source" and "case" 
df = pd.merge(gold, sys, on=["source", "case"])

# Finally, convert the labels to string
df["label"] = df["label"].astype(str)
df["pred"] = df["pred"].astype(str)


# For each source, compute the CEM-Ord score
cem_ord_scores = []
for source in df["source"].unique():
    source_df = df[df["source"] == source]
    gold_labels = source_df["label"].tolist()
    system_labels = source_df["pred"].tolist()
    cem_ord_metric = CEMOrd(gold_labels, system_labels)
    score = cem_ord_metric.evaluate()
    print(f"Source: {source}, CEM-Ord Score: {score}")



