"""Part 1: Clustering (K-Means, Hierarchical) + PCA on the Wine dataset."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

FIG = "outputs/figures/"
data = load_wine(as_frame=True)
X, y = data.data, data.target
Xs = StandardScaler().fit_transform(X)

# ---- PCA ----
pca_full = PCA().fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95) + 1)
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.axhline(0.95, color="r", ls="--", label="95% variance")
plt.xlabel("Number of components"); plt.ylabel("Cumulative explained variance")
plt.title("PCA - Cumulative Explained Variance"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG + "pca_variance.png", dpi=150); plt.close()
X2 = PCA(n_components=2, random_state=42).fit_transform(Xs)

# ---- Choose k: elbow + silhouette ----
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, marker="o"); ax[0].set_title("Elbow Method")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia (WCSS)"); ax[0].grid(alpha=.3)
ax[1].plot(list(ks), sil, marker="o", color="green"); ax[1].set_title("Silhouette Score vs k")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG + "elbow_silhouette.png", dpi=150); plt.close()
best_k_sil = list(ks)[int(np.argmax(sil))]

# ---- Final models with k=3 ----
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Xs)
hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Xs)
res = {}
for name, lab in [("K-Means", km.labels_), ("Hierarchical (Ward)", hc.labels_)]:
    res[name] = dict(silhouette=float(silhouette_score(Xs, lab)),
                     davies_bouldin=float(davies_bouldin_score(Xs, lab)),
                     adjusted_rand_index=float(adjusted_rand_score(y, lab)),
                     cluster_sizes=[int(v) for v in np.bincount(lab)])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for a, lab, t in zip(ax, [km.labels_, hc.labels_, y],
                     ["K-Means (k=3)", "Hierarchical Ward (k=3)", "True Wine Classes"]):
    s = a.scatter(X2[:, 0], X2[:, 1], c=lab, cmap="viridis", s=28, edgecolor="k", lw=.3)
    a.set_title(t); a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig(FIG + "cluster_scatter.png", dpi=150); plt.close()

Z = linkage(Xs, method="ward")
plt.figure(figsize=(10, 4.5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, show_contracted=True)
plt.title("Hierarchical Clustering Dendrogram (Ward, truncated)")
plt.xlabel("Sample (cluster size)"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig(FIG + "dendrogram.png", dpi=150); plt.close()

# cluster profiles
prof = pd.DataFrame(X).assign(cluster=km.labels_).groupby("cluster").mean().round(2)
prof.to_csv("outputs/kmeans_cluster_profiles.csv")

out = dict(n_samples=int(X.shape[0]), n_features=int(X.shape[1]),
           pca_components_95=n95,
           pca_pc1_pc2_var=[float(v) for v in pca_full.explained_variance_ratio_[:2]],
           best_k_by_silhouette=int(best_k_sil),
           silhouette_by_k={int(k): float(s) for k, s in zip(ks, sil)},
           inertia_by_k={int(k): float(s) for k, s in zip(ks, inertia)},
           models=res)
json.dump(out, open("outputs/clustering_results.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print(prof[["alcohol", "flavanoids", "color_intensity", "proline"]])
