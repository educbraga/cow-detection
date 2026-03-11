"""
Step 4 of the challenge:
Descriptive analysis of geometric features and their usability for cow identification.
Generates per-cow statistics, separability metrics, ANOVA ranking, and visualizations.
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
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description="Descriptive analysis of cow features (Step 4)")
    parser.add_argument("--input", type=str, default="outputs/reports/classification_features.csv",
                        help="Input features CSV (must have cow_id column)")
    parser.add_argument("--output-dir", type=str, default="outputs/figures",
                        help="Output directory for figures")
    parser.add_argument("--report", type=str, default="outputs/reports/feature_analysis.md",
                        help="Output report path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run train_classifier.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    if "cow_id" not in df.columns or df["cow_id"].isna().all():
        print("ERROR: CSV must have a 'cow_id' column with values.")
        sys.exit(1)

    df["cow_id"] = df["cow_id"].astype(str)
    n_cows = df["cow_id"].nunique()
    print(f"Found {n_cows} unique cows")

    # ── Identify feature columns ──
    meta_cols = ["cow_id", "image", "session_id"]
    kp_raw_cols = [c for c in df.columns if c.startswith("kp_")]
    feature_cols = [c for c in df.columns if c not in meta_cols + kp_raw_cols
                    and df[c].dtype in ["float64", "float32", "int64"]]

    angle_cols = [c for c in feature_cols if c.startswith("angle_")]
    ratio_cols = [c for c in feature_cols if c.startswith("ratio_")]
    dist_cols = [c for c in feature_cols if c.startswith("dist_")]
    selected = angle_cols + ratio_cols  # scale-invariant features

    print(f"Features: {len(angle_cols)} angles, {len(dist_cols)} distances, {len(ratio_cols)} ratios")
    print(f"Using {len(selected)} scale-invariant features for analysis")

    df_clean = df.dropna(subset=selected).copy()
    print(f"Samples after dropping NaN: {len(df_clean)} ({len(df)-len(df_clean)} dropped)")

    # ====================================================================
    # 1. DESCRIPTIVE STATISTICS (global + per cow)
    # ====================================================================
    print("\n=== Global Descriptive Statistics ===")
    desc_global = df_clean[selected].describe()
    print(desc_global.to_string())

    desc_per_cow = df_clean.groupby("cow_id")[selected].agg(["mean", "std"])
    samples_per_cow = df_clean["cow_id"].value_counts().sort_index()

    # ====================================================================
    # 2. ANOVA F-STATISTIC — ranking of discriminative features
    # ====================================================================
    print("\n=== ANOVA F-statistic (feature ranking) ===")
    anova_results = []
    for feat in selected:
        groups = [g[feat].dropna().values for _, g in df_clean.groupby("cow_id")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            anova_results.append({
                "feature": feat,
                "F_statistic": round(f_stat, 2),
                "p_value": p_val,
                "significant": "✓" if p_val < 0.05 else "✗",
            })

    anova_df = pd.DataFrame(anova_results).sort_values("F_statistic", ascending=False)
    print(anova_df.to_string(index=False))

    # ====================================================================
    # 3. SEPARABILITY — inter-class vs intra-class variance
    # ====================================================================
    print("\n=== Separability Analysis ===")
    sep_results = []
    for feat in selected:
        overall_mean = df_clean[feat].mean()
        group_stats = df_clean.groupby("cow_id")[feat].agg(["mean", "var", "count"])
        group_stats = group_stats[group_stats["count"] >= 2]

        # Inter-class variance: variance of group means
        inter_var = group_stats["mean"].var()
        # Intra-class variance: weighted average of within-group variances
        total_n = group_stats["count"].sum()
        intra_var = (group_stats["var"] * group_stats["count"]).sum() / total_n

        ratio = inter_var / (intra_var + 1e-10)
        sep_results.append({
            "feature": feat,
            "inter_class_var": round(inter_var, 4),
            "intra_class_var": round(intra_var, 4),
            "separability_ratio": round(ratio, 4),
        })

    sep_df = pd.DataFrame(sep_results).sort_values("separability_ratio", ascending=False)
    print(sep_df.to_string(index=False))

    # ====================================================================
    # 4. FIGURES
    # ====================================================================
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # Top features by separability
    top_features = sep_df.head(6)["feature"].tolist()

    # ── 4a. Violinplots per cow (top 4 features) ──
    for feat in top_features[:4]:
        fig, ax = plt.subplots(figsize=(16, 5))
        order = df_clean.groupby("cow_id")[feat].median().sort_values().index
        sns.violinplot(data=df_clean, x="cow_id", y=feat, order=order,
                       hue="cow_id", ax=ax, palette="husl", inner="box",
                       linewidth=0.5, cut=0, legend=False)
        ax.set_title(f"{feat.replace('_', ' ').title()} — Distribution per Cow",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Cow ID")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, fontsize=7)
        plt.tight_layout()
        fname = f"violinplot_{feat}_by_cow.png"
        plt.savefig(fig_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")

    # ── 4b. Separability heatmap ──
    fig, ax = plt.subplots(figsize=(12, 5))
    sep_matrix = sep_df.set_index("feature")[["separability_ratio"]].T
    sns.heatmap(sep_matrix, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                xticklabels=[f.replace("_", " ") for f in sep_matrix.columns],
                linewidths=0.5)
    ax.set_title("Feature Separability (Inter-class / Intra-class Variance)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "separability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: separability_heatmap.png")

    # ── 4c. ANOVA F-statistic bar chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    anova_sorted = anova_df.head(len(selected))
    colors = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in anova_sorted["p_value"]]
    bars = ax.barh(range(len(anova_sorted)), anova_sorted["F_statistic"].values, color=colors)
    ax.set_yticks(range(len(anova_sorted)))
    ax.set_yticklabels([f.replace("_", " ").title() for f in anova_sorted["feature"]], fontsize=8)
    ax.set_xlabel("F-statistic")
    ax.set_title("ANOVA F-statistic — Feature Discriminative Power\n(Green = significant p<0.05)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_dir / "anova_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: anova_ranking.png")

    # ── 4d. Pairplot colored by cow_id (top 4 features) ──
    if len(top_features) >= 2:
        pair_feats = top_features[:4]
        df_pair = df_clean[pair_feats + ["cow_id"]].dropna().reset_index(drop=True)
        # Subsample if too many points for readability
        if len(df_pair) > 2000:
            sampled = []
            for cid, grp in df_pair.groupby("cow_id"):
                sampled.append(grp.sample(min(len(grp), 60), random_state=42))
            df_pair = pd.concat(sampled, ignore_index=True)
        g = sns.pairplot(df_pair, hue="cow_id", diag_kind="kde",
                         plot_kws={"alpha": 0.4, "s": 15}, palette="husl",
                         height=2.5)
        g.fig.suptitle("Pairplot of Top Features — Colored by Cow ID",
                       y=1.02, fontsize=14, fontweight="bold")
        # Move legend outside
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        plt.tight_layout()
        plt.savefig(fig_dir / "pairplot_by_cow.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("Saved: pairplot_by_cow.png")

    # ── 4e. Correlation heatmap ──
    if len(selected) > 2:
        corr = df_clean[selected].corr()
        fig, ax = plt.subplots(figsize=(max(10, len(selected)*0.7),
                                        max(8, len(selected)*0.6)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=len(selected) <= 15, fmt=".2f",
                    cmap="RdBu_r", center=0, square=True, ax=ax,
                    xticklabels=[c.replace("_", " ") for c in selected],
                    yticklabels=[c.replace("_", " ") for c in selected])
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: correlation_heatmap.png")

    # ── 4f. Histograms of angles ──
    if angle_cols:
        n_angles = len(angle_cols)
        fig, axes = plt.subplots(1, n_angles, figsize=(5 * n_angles, 4))
        if n_angles == 1:
            axes = [axes]
        for ax, col in zip(axes, angle_cols):
            data = df_clean[col].dropna()
            ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
            ax.set_title(col.replace("angle_", "").replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Degrees")
            ax.set_ylabel("Count")
        plt.suptitle("Distribution of Angles", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir / "histograms_angles.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: histograms_angles.png")

    # ── 4g. Histograms of ratios ──
    if ratio_cols:
        n_ratios = len(ratio_cols)
        n_rows = (n_ratios + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes_flat = axes.flatten() if n_ratios > 3 else ([axes] if n_ratios == 1 else axes.flatten())
        for i, col in enumerate(ratio_cols):
            data = df_clean[col].dropna()
            axes_flat[i].hist(data, bins=30, edgecolor="black", alpha=0.7, color="#55A868")
            axes_flat[i].set_title(col.replace("ratio_", "").replace("_", " ").title(), fontsize=9)
            axes_flat[i].set_xlabel("Ratio")
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Distribution of Ratios", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_dir / "histograms_ratios.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: histograms_ratios.png")

    # ====================================================================
    # 5. GENERATE REPORT
    # ====================================================================
    with open(report_path, "w") as f:
        f.write("# Análise Descritiva das Features — Identificação de Vacas\n\n")

        f.write("## 1. Visão Geral do Dataset\n\n")
        f.write(f"- **Total de amostras**: {len(df_clean)}\n")
        f.write(f"- **Vacas (classes)**: {n_cows}\n")
        f.write(f"- **Features utilizadas**: {len(selected)} (scale-invariant)\n")
        f.write(f"  - Ângulos: {len(angle_cols)}\n")
        f.write(f"  - Ratios: {len(ratio_cols)}\n\n")

        f.write("### Amostras por vaca\n\n")
        f.write("| Cow ID | Amostras |\n|---|---|\n")
        for cow_id, count in samples_per_cow.items():
            f.write(f"| {cow_id} | {count} |\n")
        f.write("\n")

        f.write("## 2. Estatísticas Descritivas Globais\n\n")
        f.write("```\n")
        f.write(desc_global.to_string())
        f.write("\n```\n\n")

        # ANOVA ranking
        f.write("## 3. Ranking de Features por Poder Discriminativo (ANOVA)\n\n")
        f.write("A tabela abaixo mostra o **F-statistic** da ANOVA one-way para cada feature, ")
        f.write("indicando quão bem essa feature separa as vacas. Quanto maior o valor, ")
        f.write("mais discriminativa é a feature.\n\n")
        f.write("| Feature | F-statistic | p-value | Significante |\n")
        f.write("|---|---|---|---|\n")
        for _, row in anova_df.iterrows():
            feat_name = row["feature"].replace("_", " ").title()
            p_str = f"{row['p_value']:.2e}" if row["p_value"] < 0.001 else f"{row['p_value']:.4f}"
            f.write(f"| {feat_name} | {row['F_statistic']:.2f} | {p_str} | {row['significant']} |\n")
        f.write("\n")

        # Separability table
        f.write("## 4. Separabilidade (Variância Inter-classe vs Intra-classe)\n\n")
        f.write("A **razão de separabilidade** mede quanto a variação entre vacas é maior ")
        f.write("que a variação dentro de uma mesma vaca. Valores > 1 indicam que a feature ")
        f.write("varia mais entre vacas do que dentro de cada vaca.\n\n")
        f.write("| Feature | Var. Inter-classe | Var. Intra-classe | Razão de Separabilidade |\n")
        f.write("|---|---|---|---|\n")
        for _, row in sep_df.iterrows():
            feat_name = row["feature"].replace("_", " ").title()
            f.write(f"| {feat_name} | {row['inter_class_var']:.4f} | {row['intra_class_var']:.4f} | {row['separability_ratio']:.4f} |\n")
        f.write("\n")

        # Correlation highlights
        f.write("## 5. Correlação entre Features\n\n")
        if len(selected) > 2:
            corr_matrix = df_clean[selected].corr()
            high_corr = []
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    r = corr_matrix.iloc[i, j]
                    if abs(r) > 0.7:
                        high_corr.append((selected[i], selected[j], round(r, 3)))

            if high_corr:
                f.write("Pares de features com alta correlação (|r| > 0.7):\n\n")
                f.write("| Feature A | Feature B | Correlação |\n|---|---|---|\n")
                for a, b, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                    f.write(f"| {a.replace('_',' ').title()} | {b.replace('_',' ').title()} | {r:.3f} |\n")
                f.write("\n")
            else:
                f.write("Nenhum par de features com correlação > 0.7.\n\n")

        # Conclusions
        f.write("## 6. Conclusões\n\n")

        best_feats = anova_df.head(3)["feature"].tolist()
        worst_feats = anova_df.tail(3)["feature"].tolist()

        f.write("### Features mais discriminativas\n\n")
        f.write("As features com maior poder discriminativo (maior F-statistic na ANOVA) são:\n\n")
        for i, feat in enumerate(best_feats, 1):
            row = sep_df[sep_df["feature"] == feat].iloc[0]
            f.write(f"{i}. **{feat.replace('_', ' ').title()}** — ")
            f.write(f"F={anova_df[anova_df['feature']==feat].iloc[0]['F_statistic']:.1f}, ")
            f.write(f"razão de separabilidade={row['separability_ratio']:.3f}\n")
        f.write("\n")

        f.write("### Features menos discriminativas\n\n")
        f.write("As features com menor poder discriminativo são:\n\n")
        for i, feat in enumerate(worst_feats, 1):
            f_val = anova_df[anova_df["feature"]==feat].iloc[0]["F_statistic"]
            f.write(f"{i}. **{feat.replace('_', ' ').title()}** — F={f_val:.1f}\n")
        f.write("\n")

        f.write("### Avaliação geral da usabilidade para identificação\n\n")

        n_significant = len(anova_df[anova_df["significant"] == "✓"])
        n_total = len(anova_df)
        best_sep = sep_df.iloc[0]

        f.write(f"- **{n_significant}/{n_total} features** apresentam diferença estatisticamente ")
        f.write(f"significativa entre vacas (p < 0.05).\n")
        f.write(f"- A feature mais discriminativa ({best_sep['feature'].replace('_',' ').title()}) ")
        f.write(f"tem razão de separabilidade de **{best_sep['separability_ratio']:.3f}**.\n")

        if best_sep["separability_ratio"] > 1.0:
            f.write("- A razão de separabilidade > 1 indica que existe variação inter-classe ")
            f.write("superior à intra-classe, sugerindo que as features **capturam diferenças ")
            f.write("individuais entre vacas**.\n")
        else:
            f.write("- A razão de separabilidade < 1 indica que a variação dentro de cada vaca ")
            f.write("é comparável à variação entre vacas. As features geométricas, por si só, ")
            f.write("são **limitadas para identificação individual**, embora ainda superiores ao acaso.\n")

        f.write("- As **features de ângulo** tendem a ser mais discriminativas que os ratios, ")
        f.write("pois capturam a geometria relativa dos keypoints sem depender de escala.\n")
        f.write("- A **limitação principal** é que as variações intra-classe (mesma vaca em diferentes ")
        f.write("posições) podem ser grandes, dificultando a separação entre indivíduos.\n\n")

        f.write("## 7. Gráficos Gerados\n\n")
        for fig_file in sorted(fig_dir.glob("*.png")):
            f.write(f"- `{fig_file.name}`\n")
        f.write("\n")

    print(f"\nReport saved to: {report_path}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
