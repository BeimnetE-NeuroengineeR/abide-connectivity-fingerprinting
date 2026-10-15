import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print("Loading ABIDE dataset...")
abide = datasets.fetch_abide_pcp(
    derivatives=["rois_cc200"],
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    quality_checked=True,
    n_subjects=None
)

pheno = pd.DataFrame(abide.phenotypic)
y = (pheno["DX_GROUP"].values == 1).astype(int)
sites = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in pheno["SITE_ID"]])

# Extract Functional Connectivity
conn = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
X = np.nan_to_num(conn.fit_transform(abide.rois_cc200))

print(f"Subjects: {len(X)} | Sites: {len(set(sites))}")

# Calculate Within-Group vs. Between-Group FC Cosine Similarity
sim_matrix = cosine_similarity(X)

# Mask for same-diagnosis vs different-diagnosis pairs
same_dx = (y[:, None] == y[None, :]) & (~np.eye(len(y), dtype=bool))
diff_dx = (y[:, None] != y[None, :])

# Mask for same-site vs different-site pairs
same_site = (sites[:, None] == sites[None, :]) & (~np.eye(len(sites), dtype=bool))
diff_site = (sites[:, None] != sites[None, :])

mean_sim_same_site = sim_matrix[same_site].mean()
mean_sim_diff_site = sim_matrix[diff_site].mean()

mean_sim_same_dx = sim_matrix[same_dx].mean()
mean_sim_diff_dx = sim_matrix[diff_dx].mean()

print("\n--- Functional Connectivity Similarity Analysis ---")
print(f"Mean Cosine Similarity (Same Site): {mean_sim_same_site:.4f}")
print(f"Mean Cosine Similarity (Different Sites): {mean_sim_diff_site:.4f}")
print(f"Mean Cosine Similarity (Same Diagnosis): {mean_sim_same_dx:.4f}")
print(f"Mean Cosine Similarity (Different Diagnosis): {mean_sim_diff_dx:.4f}")

# Save Summary Table
df_res = pd.DataFrame([
    {"Comparison": "Same Site", "Cosine_Similarity": mean_sim_same_site},
    {"Comparison": "Different Sites", "Cosine_Similarity": mean_sim_diff_site},
    {"Comparison": "Same Diagnosis", "Cosine_Similarity": mean_sim_same_dx},
    {"Comparison": "Different Diagnosis", "Cosine_Similarity": mean_sim_diff_dx}
])
df_res.to_csv("reports/fingerprint_similarity.csv", index=False)

# Bar chart output
plt.figure(figsize=(8, 4.5))
plt.bar(["Same Site", "Diff Site", "Same Dx", "Diff Dx"], 
        [mean_sim_same_site, mean_sim_diff_site, mean_sim_same_dx, mean_sim_diff_dx],
        color=['#2b5c8f', '#41b6c4', '#c7e9b4', '#7fcdbb'])
plt.ylabel("Mean Cosine Similarity")
plt.title("Functional Connectivity Profile Similarity across Sites & Diagnoses")
plt.tight_layout()
plt.savefig("reports/similarity_breakdown.png", dpi=150)
print("Analysis complete. Plot saved to reports/similarity_breakdown.png")