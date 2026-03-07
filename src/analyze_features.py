"""
Descriptive analysis of geometric features extracted from cow keypoints.
Generates statistics, histograms, correlation heatmaps, and pairplots.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Analyze cow pose features")
    parser.add_argument("--input", type=str, default="data/processed/features.csv", help="Input features CSV")
    parser.add_argument("--output-dir", type=str, default="outputs/figures", help="Output directory for figures")
    parser.add_argument("--report", type=str, default="outputs/reports/feature_analysis.md", help="Output report path")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run extract_features.py first.")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Identify feature columns (numeric, not metadata)
    meta_cols = ["image", "cow_id", "station", "cam", "date", "time", "n_keypoints_detected"]
    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ["float64", "float32", "int64"]]
    
    angle_cols = [c for c in feature_cols if c.startswith("angle_")]
    dist_cols = [c for c in feature_cols if c.startswith("dist_")]
    ratio_cols = [c for c in feature_cols if c.startswith("ratio_")]
    
    print(f"Features: {len(angle_cols)} angles, {len(dist_cols)} distances, {len(ratio_cols)} ratios")
    
    # ---- 1. Descriptive Statistics ----
    desc = df[feature_cols].describe()
    print("\n=== Descriptive Statistics ===")
    print(desc.to_string())
    
    # ---- 2. Histograms of Angles ----
    if angle_cols:
        n_angles = len(angle_cols)
        fig, axes = plt.subplots(1, n_angles, figsize=(5 * n_angles, 4))
        if n_angles == 1:
            axes = [axes]
        for ax, col in zip(axes, angle_cols):
            data = df[col].dropna()
            ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
            ax.set_title(col.replace("angle_", "").replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Degrees")
            ax.set_ylabel("Count")
        plt.suptitle("Distribution of Angles", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir / "histograms_angles.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: histograms_angles.png")
    
    # ---- 3. Histograms of Distances ----
    if dist_cols:
        n_dists = len(dist_cols)
        n_rows = (n_dists + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_dists > 3 else ([axes] if n_dists == 1 else axes.flatten())
        for i, col in enumerate(dist_cols):
            data = df[col].dropna()
            axes[i].hist(data, bins=30, edgecolor="black", alpha=0.7, color="#55A868")
            axes[i].set_title(col.replace("dist_", "").replace("_", " ").title(), fontsize=9)
            axes[i].set_xlabel("Pixels")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Distribution of Distances", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir / "histograms_distances.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: histograms_distances.png")
    
    # ---- 4. Correlation Heatmap ----
    if len(feature_cols) > 2:
        # Select features with enough non-null values
        valid_features = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.3]
        if len(valid_features) > 2:
            corr = df[valid_features].corr()
            fig, ax = plt.subplots(figsize=(max(12, len(valid_features) * 0.6), max(10, len(valid_features) * 0.5)))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=len(valid_features) <= 15, fmt=".2f",
                       cmap="RdBu_r", center=0, square=True, ax=ax,
                       xticklabels=[c.replace("_", " ") for c in valid_features],
                       yticklabels=[c.replace("_", " ") for c in valid_features])
            ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()
            plt.savefig(fig_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved: correlation_heatmap.png")
    
    # ---- 5. Distribution by Station (proxy for cow variation) ----
    if "station" in df.columns and df["station"].notna().any():
        top_stations = df["station"].value_counts().head(8).index.tolist()
        df_top = df[df["station"].isin(top_stations)]
        
        # Boxplot of angles by station
        if angle_cols and len(df_top) > 0:
            for col in angle_cols[:3]:  # Top 3 angles
                fig, ax = plt.subplots(figsize=(10, 5))
                df_plot = df_top[[col, "station"]].dropna()
                if len(df_plot) > 0:
                    sns.boxplot(data=df_plot, x="station", y=col, ax=ax, palette="Set2")
                    ax.set_title(f"{col.replace('_', ' ').title()} by Station", fontsize=12)
                    ax.set_xlabel("Station")
                    ax.set_ylabel("Degrees")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    fname = f"boxplot_{col}_by_station.png"
                    plt.savefig(fig_dir / fname, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"Saved: {fname}")
    
    # ---- 6. Pairplot of Key Features ----
    key_features = angle_cols[:4] if len(angle_cols) >= 4 else angle_cols + ratio_cols[:4 - len(angle_cols)]
    key_features = [c for c in key_features if df[c].notna().sum() > 10]
    
    if len(key_features) >= 2:
        df_pair = df[key_features].dropna()
        if len(df_pair) > 5:
            g = sns.pairplot(df_pair, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
            g.fig.suptitle("Pairplot of Key Angle Features", y=1.02, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_dir / "pairplot_angles.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved: pairplot_angles.png")
    
    # ---- 7. Feature Variability Analysis ----
    if "station" in df.columns and df["station"].notna().any():
        valid_feats = [c for c in angle_cols + ratio_cols if df[c].notna().sum() > 20]
        if valid_feats:
            cv_data = []
            for feat in valid_feats:
                overall_std = df[feat].std()
                overall_mean = df[feat].mean()
                cv = overall_std / abs(overall_mean) if abs(overall_mean) > 1e-8 else np.nan
                
                # Inter-station variance
                group_means = df.groupby("station")[feat].mean()
                inter_var = group_means.std() if len(group_means) > 1 else np.nan
                
                cv_data.append({
                    "feature": feat,
                    "mean": overall_mean,
                    "std": overall_std,
                    "cv": cv,
                    "inter_station_std": inter_var,
                })
            cv_df = pd.DataFrame(cv_data)
            print("\n=== Feature Variability ===")
            print(cv_df.to_string(index=False))
    
    # ---- 8. Generate Report ----
    with open(report_path, "w") as f:
        f.write("# Feature Analysis Report\n\n")
        f.write(f"**Total samples:** {len(df)}\n")
        f.write(f"**Feature columns:** {len(feature_cols)}\n")
        f.write(f"  - Angles: {len(angle_cols)}\n")
        f.write(f"  - Distances: {len(dist_cols)}\n")
        f.write(f"  - Ratios: {len(ratio_cols)}\n\n")
        
        if "station" in df.columns:
            f.write(f"**Unique stations:** {df['station'].nunique()}\n")
            f.write(f"**Images per station:**\n\n")
            f.write(df["station"].value_counts().to_string() + "\n\n")
        
        if "cow_id" in df.columns and df["cow_id"].notna().any():
            f.write(f"**Unique cow_ids:** {df['cow_id'].nunique()}\n\n")
        
        f.write("## Descriptive Statistics\n\n")
        f.write("```\n")
        f.write(desc.to_string())
        f.write("\n```\n\n")
        
        f.write("## Generated Figures\n\n")
        for fig_file in sorted(fig_dir.glob("*.png")):
            f.write(f"- `{fig_file.name}`\n")
        
        f.write("\n## Key Observations\n\n")
        f.write("- Features with high coefficient of variation (CV) may be useful for individual identification.\n")
        f.write("- Strongly correlated features (|r| > 0.8) may be redundant and candidates for removal.\n")
        f.write("- Angle features tend to be more invariant to scale/distance than raw distances.\n")
    
    print(f"\nReport saved to: {report_path}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
