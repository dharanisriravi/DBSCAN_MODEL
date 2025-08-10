
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def _choose_numeric_columns(df):
    # favor common names; fallback to any numeric cols excluding ID-like
    prefer = ["TotalSpend", "VisitFrequency", "CategoriesBought", "Total_Spend", "Visit_Frequency"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    chosen = [c for c in prefer if c in numeric_cols]
    if len(chosen) >= 2:
        return chosen
    # fallback: use top 3 numeric columns excluding ID strings / obvious index columns
    filtered = [c for c in numeric_cols if "id" not in c.lower()]
    return filtered[:3] if filtered else numeric_cols

def run_dbscan_and_prepare(csv_path, eps=0.5, min_samples=5):
    df = pd.read_csv(csv_path)
    # minimal validation
    if df.shape[0] < 3:
        raise ValueError("CSV must contain at least 3 rows of data.")

    # identify an ID column or create one
    id_col = None
    for col in df.columns:
        if col.lower() in ("customerid", "customer_id", "id"):
            id_col = col
            break
    if id_col is None:
        id_col = "CustomerID"
        df[id_col] = [f"CUST_{i+1}" for i in range(len(df))]

    # choose numeric features for clustering
    num_cols = _choose_numeric_columns(df)
    if len(num_cols) < 2:
        raise ValueError("Need at least two numeric features for clustering. Detected numeric columns: " + ", ".join(df.select_dtypes(include=[np.number]).columns))

    X = df[num_cols].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(Xs)
    df["cluster"] = labels

    # 2D projection for plotting using PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    df["_x"] = coords[:, 0]
    df["_y"] = coords[:, 1]

    # Build cluster datasets for chart: group by cluster label
    clusters_for_chart = []
    unique_labels = sorted(df["cluster"].unique())
    for lbl in unique_labels:
        sub = df[df["cluster"] == lbl]
        points = []
        for _, r in sub.iterrows():
            points.append({
                "x": float(r["_x"]),
                "y": float(r["_y"]),
                "id": str(r[id_col]),
                # include some hover text data
                "total_spend": float(r[num_cols[0]]) if num_cols and num_cols[0] in r else None,
                "visit_freq": float(r[num_cols[1]]) if len(num_cols) > 1 and num_cols[1] in r else None
            })
        clusters_for_chart.append({
            "label": str(lbl),
            "points": points,
            "size": len(points)
        })

    # Cluster summary table
    summary_rows = []
    grouped = df.groupby("cluster")
    for lbl, g in grouped:
        row = {
            "cluster": int(lbl),
            "size": int(len(g)),
        }
        # compute means for numeric columns
        for c in num_cols:
            row[f"mean_{c}"] = round(float(g[c].mean()), 3)
        # example insights: high/low for first two metrics (relative)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(by="cluster").reset_index(drop=True)

    # preview (first 8 rows) as html table (safe)
    preview_html = df[[id_col] + num_cols + ["cluster"]].head(8).to_html(classes="table table-sm table-striped", index=False, float_format="{:0.3f}".format)

    return {
        "clusters_for_chart": clusters_for_chart,
        "summary_table": summary_df,
        "preview_html": preview_html
    }
