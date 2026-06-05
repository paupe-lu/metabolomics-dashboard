"""
Streamlit dashboard for mouse untargeted metabolomics.

Expected input Excel sheets:
- "Normalized results": first column = metabolite names, remaining columns = samples
- "Metadata": contains Mice number, Group, Gender, Backgound, Date, Sample name

Run:
    streamlit run streamlit_metabolomics_dashboard_v2.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


st.set_page_config(
    page_title="Mouse untargeted metabolomics dashboard",
    layout="wide",
)

GROUP_ORDER = [
    "GF",
    "SPF",
    "SPF Before Abx",
    "SPF Abx",
    "SPF After Abx",
]

ABX_ORDER = ["SPF Before Abx", "SPF Abx", "SPF After Abx"]

EXPERIMENT_ORDER = ["GF vs SPF", "Antibiotics experiment"]

CLASS_ORDER = [
    "Amino acids and derivatives",
    "Tryptophan, indoles and kynurenine pathway",
    "Purines and purine nucleotides",
    "Pyrimidines and pyrimidine nucleotides",
    "Central carbon metabolism",
    "Organic acids",
    "Vitamins and cofactors",
    "Redox, sulfur and methyl metabolism",
    "Carnitines and lipid-related",
    "Neuroactive / aromatic amines",
    "Other / unclassified",
]







def add_white_points_on_boxes(
    fig,
    data,
    x_col,
    y_col,
    sample_col="MetabolomicsSample",
    point_size=7,
):
    """
    Add white sample dots as separate scatter traces centered on the same
    categorical x positions as the boxplots. This preserves the original
    boxplot colors.
    """
    import plotly.graph_objects as go

    # Use the actual category order as shown in the plotted dataframe.
    categories = list(pd.unique(data[x_col].dropna()))

    for category in categories:
        sub = data.loc[data[x_col] == category].copy()
        if sub.empty:
            continue

        hover_text = (
            sub[sample_col].astype(str)
            if sample_col in sub.columns
            else pd.Series([""] * len(sub))
        )

        fig.add_trace(
            go.Scatter(
                x=[category] * len(sub),
                y=sub[y_col],
                mode="markers",
                marker=dict(
                    color="white",
                    size=point_size,
                    line=dict(color="black", width=1),
                    opacity=0.95,
                ),
                name="Samples",
                text=hover_text,
                hovertemplate=(
                    f"{x_col}: %{{x}}<br>"
                    f"{y_col}: %{{y:.4g}}<br>"
                    "%{text}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(boxmode="overlay")
    return fig


# -----------------------------
# Data loading and cleaning
# -----------------------------

@st.cache_data
def load_workbook(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metabolomics and metadata sheets from the uploaded Excel file."""
    met = pd.read_excel(uploaded_file, sheet_name="Normalized results")
    meta = pd.read_excel(uploaded_file, sheet_name="Metadata")
    return met, meta


def normalize_group_name(value: object) -> str:
    """Normalize group names from metadata or sample labels."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()

    replacements = {
        "SPF Before ABx": "SPF Before Abx",
        "SPF Before Abx": "SPF Before Abx",
        "SPF Abx": "SPF Abx",
        "SPF aBx": "SPF Abx",
        "SPF ABx": "SPF Abx",
        "SPF After ABx": "SPF After Abx",
        "SPF After Abx": "SPF After Abx",
        "GF": "GF",
        "SPF": "SPF",
    }

    return replacements.get(text, text)


def detect_group_from_metabolomics_sample(sample_name: str) -> str:
    """Detect biological group from metabolomics column name."""
    if "Before_Abx" in sample_name:
        return "SPF Before Abx"
    if "After_Abx" in sample_name:
        return "SPF After Abx"
    if "SPF_Abx" in sample_name:
        return "SPF Abx"
    if "_GF" in sample_name:
        return "GF"
    if "_SPF" in sample_name:
        return "SPF"
    return "Unassigned"


def detect_experiment_from_group(group: str) -> str:
    """Assign experiment based on detected/cleaned group."""
    if group in {"GF", "SPF"}:
        return "GF vs SPF"
    if group in {"SPF Before Abx", "SPF Abx", "SPF After Abx"}:
        return "Antibiotics experiment"
    return "Unassigned"


def extract_mouse_number(sample_name: str) -> int | None:
    """Extract mouse number from metabolomics sample name."""
    match = re.search(r"N(\d+)", sample_name)
    if match is None:
        return None
    return int(match.group(1))


def order_groups(series: pd.Series) -> pd.Categorical:
    """Return ordered categorical group labels."""
    observed = [x for x in series.dropna().unique().tolist() if x not in GROUP_ORDER]
    categories = GROUP_ORDER + sorted(observed)
    return pd.Categorical(series, categories=categories, ordered=True)


def classify_metabolite(name: object) -> str:
    """Rule-based metabolite classification from the actual metabolite names in this dataset."""
    if pd.isna(name):
        return "Other / unclassified"

    x = str(name).lower()

    amino_terms = [
        "alanine", "arginine", "asparagine", "aspartic", "cystathionine", "cystine",
        "glutamic", "glutamine", "glycine", "histidine", "isoleucine", "leucine",
        "lysine", "methionine", "ornithine", "phenylalanine", "proline", "serine",
        "threonine", "tyrosine", "valine", "acetylglutamic", "acetyl-l-aspartic",
        "acetylserine", "phosphoserine", "pyroglutamic", "ketoleucine",
        "n-formyl-l-methionine", "4-hydroxy-l-glutamic", "4-hydroxyproline",
    ]
    if any(term in x for term in amino_terms):
        return "Amino acids and derivatives"

    trp_terms = [
        "tryptophan", "indole", "kynuren", "xanthuren", "anthranilic",
        "picolinic", "quinolinic", "quinaldic", "tryptamine", "tryptophol",
        "serotonin", "melatonin",
    ]
    if any(term in x for term in trp_terms):
        return "Tryptophan, indoles and kynurenine pathway"

    purine_terms = [
        "adenine", "adenosine", "amp", "adp", "atp", "aicar", "guanine", "guanosine",
        "gmp", "gdp", "gtp", "inosine", "inosinic", "uric acid",
    ]
    if any(term in x for term in purine_terms):
        return "Purines and purine nucleotides"

    pyrimidine_terms = [
        "cytidine", "cytosine", "cdp", "ctp", "dump", "orotic", "thymidine", "thymine",
        "uracil", "uridine", "udp", "ump", "utp",
    ]
    if any(term in x for term in pyrimidine_terms):
        return "Pyrimidines and pyrimidine nucleotides"

    central_terms = [
        "glucose", "fructose", "galactose", "mannose", "ribose", "sedoheptulose",
        "erythrose", "phosphogluconic", "glycerate 3-phosphate", "glycerol 3-phosphate",
        "fructose 1,6", "glucose 6-phosphate", "phosphoenolpyruvic", "pyruvic",
        "lactic", "fumaric", "malic", "oxalacetic", "aconitic", "citric", "isocitric",
        "oxoglutaric", "succinic",
    ]
    if any(term in x for term in central_terms):
        return "Central carbon metabolism"

    vitamin_terms = [
        "biotin", "fad", "flavin", "nad", "nadh", "nadp", "nadph", "niacinamide",
        "nicotinic", "pantothenic", "pyridox", "riboflavin", "thiamine", "coa",
    ]
    if any(term in x for term in vitamin_terms):
        return "Vitamins and cofactors"

    redox_terms = [
        "glutathione", "cysteine", "cystathionine", "homocysteine", "taurine", "thiosulfate",
        "betaine", "sarcosine", "choline", "methylmalonic",
    ]
    if any(term in x for term in redox_terms):
        return "Redox, sulfur and methyl metabolism"

    lipid_terms = ["carnitine", "acetylcarnitine", "o-phosphoethanolamine"]
    if any(term in x for term in lipid_terms):
        return "Carnitines and lipid-related"

    neuro_terms = ["creatine", "creatinine", "acetylserotonin"]
    if any(term in x for term in neuro_terms):
        return "Neuroactive / aromatic amines"

    organic_acid_terms = [
        "acid", "itaconic", "malonic", "oxalic", "maleic", "glucaric", "gluconic",
        "glutaric", "ribonic", "threonic",
    ]
    if any(term in x for term in organic_acid_terms):
        return "Organic acids"

    return "Other / unclassified"


def add_metabolite_classes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MetaboliteClass"] = out["Metabolite"].map(classify_metabolite)
    out["MetaboliteClass"] = pd.Categorical(
        out["MetaboliteClass"], categories=CLASS_ORDER, ordered=True
    )
    return out


def build_sample_map(met: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Map metabolomics sample columns to metadata rows using mouse number + group."""
    metabolite_col = met.columns[0]
    sample_cols = [c for c in met.columns if c != metabolite_col]

    sample_map = pd.DataFrame({"MetabolomicsSample": sample_cols})
    sample_map["Mouse"] = sample_map["MetabolomicsSample"].map(extract_mouse_number)
    sample_map["DetectedGroup"] = sample_map["MetabolomicsSample"].map(detect_group_from_metabolomics_sample)
    sample_map["DetectedExperiment"] = sample_map["DetectedGroup"].map(detect_experiment_from_group)

    meta_clean = meta.copy()
    meta_clean["Mouse"] = meta_clean["Mice number"]
    meta_clean["GroupClean"] = meta_clean["Group"].map(normalize_group_name)
    meta_clean["ExperimentClean"] = meta_clean["GroupClean"].map(detect_experiment_from_group)

    merged = sample_map.merge(
        meta_clean,
        how="left",
        left_on=["Mouse", "DetectedGroup"],
        right_on=["Mouse", "GroupClean"],
        validate="many_to_one",
    )

    merged["FinalGroup"] = merged["GroupClean"].fillna(merged["DetectedGroup"])
    merged["FinalExperiment"] = merged["ExperimentClean"].fillna(merged["DetectedExperiment"])
    merged["MatchStatus"] = np.where(merged["Sample name"].notna(), "Matched", "Unmatched")
    merged["FinalGroup"] = order_groups(merged["FinalGroup"])
    merged["FinalExperiment"] = pd.Categorical(
        merged["FinalExperiment"], categories=EXPERIMENT_ORDER + ["Unassigned"], ordered=True
    )

    return merged.sort_values(["FinalExperiment", "FinalGroup", "Mouse", "MetabolomicsSample"])


def make_long_data(met: pd.DataFrame, sample_map: pd.DataFrame) -> pd.DataFrame:
    """Convert wide metabolomics matrix to long format and attach metadata."""
    metabolite_col = met.columns[0]

    long_df = met.rename(columns={metabolite_col: "Metabolite"}).melt(
        id_vars="Metabolite",
        var_name="MetabolomicsSample",
        value_name="Intensity",
    )
    long_df = long_df.merge(sample_map, on="MetabolomicsSample", how="left")
    long_df = add_metabolite_classes(long_df)
    long_df["FinalGroup"] = order_groups(long_df["FinalGroup"])
    return long_df


def make_matrix(met: pd.DataFrame, log_transform: bool = True) -> pd.DataFrame:
    """Return metabolite x sample matrix."""
    metabolite_col = met.columns[0]
    matrix = met.rename(columns={metabolite_col: "Metabolite"}).set_index("Metabolite")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if log_transform:
        matrix = np.log10(matrix + 1)

    return matrix


def ordered_unique(values: Iterable) -> list[str]:
    """Return ordered unique group labels respecting biological order."""
    clean = [str(v) for v in pd.Series(values).dropna().unique()]
    ordered = [g for g in GROUP_ORDER if g in clean]
    remaining = sorted([g for g in clean if g not in ordered])
    return ordered + remaining


def ordered_samples(sample_map: pd.DataFrame) -> list[str]:
    return sample_map.sort_values(["FinalExperiment", "FinalGroup", "Mouse", "MetabolomicsSample"])[
        "MetabolomicsSample"
    ].tolist()


def differential_test(
    long_df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    log_transform: bool = True,
) -> pd.DataFrame:
    """Welch t-test metabolite-wise between two groups."""
    df = long_df[long_df[group_col].astype(str).isin([group_a, group_b])].copy()

    if log_transform:
        df["Value"] = np.log10(df["Intensity"] + 1)
        effect_label = "difference_log10_mean"
    else:
        df["Value"] = df["Intensity"]
        effect_label = "difference_mean"

    results = []

    for metabolite, sub in df.groupby("Metabolite"):
        a = sub.loc[sub[group_col].astype(str) == group_a, "Value"].dropna()
        b = sub.loc[sub[group_col].astype(str) == group_b, "Value"].dropna()

        if len(a) < 2 or len(b) < 2:
            p_value = np.nan
            statistic = np.nan
        else:
            statistic, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

        raw_a = sub.loc[sub[group_col].astype(str) == group_a, "Intensity"].dropna()
        raw_b = sub.loc[sub[group_col].astype(str) == group_b, "Intensity"].dropna()
        pseudo = 1.0
        log2fc = np.log2((raw_b.mean() + pseudo) / (raw_a.mean() + pseudo)) if len(raw_a) and len(raw_b) else np.nan

        results.append(
            {
                "Metabolite": metabolite,
                "MetaboliteClass": classify_metabolite(metabolite),
                f"mean_{group_a}": a.mean(),
                f"mean_{group_b}": b.mean(),
                effect_label: b.mean() - a.mean(),
                "log2FC_raw_mean_with_pseudocount1": log2fc,
                "t_statistic": statistic,
                "p_value": p_value,
                "n_reference": len(a),
                "n_comparison": len(b),
            }
        )

    out = pd.DataFrame(results)
    valid = out["p_value"].notna()

    out["FDR"] = np.nan
    if valid.any():
        out.loc[valid, "FDR"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]

    out["minus_log10_p"] = -np.log10(out["p_value"])
    out = out.sort_values("p_value", na_position="last")

    return out


def compute_pca(matrix: pd.DataFrame, sample_map: pd.DataFrame, n_components: int = 5):
    selected_samples = ordered_samples(sample_map)
    matrix = matrix[selected_samples]
    matrix_t = matrix.T
    matrix_t = matrix_t.fillna(matrix_t.median(axis=0))

    scaled = StandardScaler().fit_transform(matrix_t)
    pca = PCA(n_components=min(n_components, matrix_t.shape[0], matrix_t.shape[1]))
    coords = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"], index=matrix_t.index)
    pca_df = pca_df.reset_index().rename(columns={"index": "MetabolomicsSample"})
    pca_df = pca_df.merge(sample_map, on="MetabolomicsSample", how="left")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=matrix_t.columns,
        columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])],
    ).reset_index().rename(columns={"index": "Metabolite"})
    loadings = add_metabolite_classes(loadings)

    return pca, pca_df, loadings


def spearman_correlations(matrix: pd.DataFrame, selected_metabolite: str) -> pd.DataFrame:
    """Correlate one metabolite against all others across samples."""
    if selected_metabolite not in matrix.index:
        return pd.DataFrame()

    x = matrix.loc[selected_metabolite]
    rows = []
    for met_name, y in matrix.iterrows():
        if met_name == selected_metabolite:
            continue
        tmp = pd.concat([x, y], axis=1).dropna()
        tmp.columns = ["x", "y"]
        if tmp.shape[0] < 4 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = stats.spearmanr(tmp["x"], tmp["y"])
        rows.append(
            {
                "Metabolite": met_name,
                "MetaboliteClass": classify_metabolite(met_name),
                "spearman_rho": rho,
                "p_value": p_value,
                "n": tmp.shape[0],
            }
        )

    out = pd.DataFrame(rows)
    valid = out["p_value"].notna()
    out["FDR"] = np.nan
    if valid.any():
        out.loc[valid, "FDR"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
    out["abs_rho"] = out["spearman_rho"].abs()
    return out.sort_values("abs_rho", ascending=False, na_position="last")


def recovery_table(long_df: pd.DataFrame, log_transform: bool = True) -> pd.DataFrame:
    """Compute antibiotic recovery score from group means."""
    df = long_df[long_df["FinalGroup"].astype(str).isin(ABX_ORDER)].copy()
    df["Value"] = np.log10(df["Intensity"] + 1) if log_transform else df["Intensity"]
    means = (
        df.groupby(["Metabolite", "FinalGroup"], observed=False)["Value"]
        .mean()
        .reset_index()
        .pivot(index="Metabolite", columns="FinalGroup", values="Value")
    )
    for col in ABX_ORDER:
        if col not in means.columns:
            means[col] = np.nan
    means = means[ABX_ORDER]
    means["Before_minus_Abx"] = means["SPF Before Abx"] - means["SPF Abx"]
    means["After_minus_Abx"] = means["SPF After Abx"] - means["SPF Abx"]
    means["RecoveryScore"] = means["After_minus_Abx"] / means["Before_minus_Abx"]
    means["MetaboliteClass"] = [classify_metabolite(x) for x in means.index]
    return means.reset_index().sort_values("RecoveryScore", ascending=False, na_position="last")


# -----------------------------
# Sidebar and data setup
# -----------------------------

st.title("Mouse untargeted metabolomics dashboard — v6 fixed box points")
st.info("v4 fixed dots: volcano/differential analysis, PCA loadings, metabolite class explorer, correlation explorer, and antibiotic recovery are included. Antibiotics order is fixed as SPF Before Abx → SPF Abx → SPF After Abx.")

uploaded = st.sidebar.file_uploader(
    "Upload metabolomics Excel file",
    type=["xlsx"],
)

default_path = Path("untargeted_GF_SPF_background.xlsx")

if uploaded is None and default_path.exists():
    uploaded_source = default_path
    st.sidebar.info("Using local file: untargeted_GF_SPF_background.xlsx")
elif uploaded is not None:
    uploaded_source = uploaded
else:
    st.warning("Upload the Excel file to start.")
    st.stop()

met, meta = load_workbook(uploaded_source)
sample_map = build_sample_map(met, meta)
long_df = make_long_data(met, sample_map)

metabolites = sorted(long_df["Metabolite"].dropna().unique())
experiments = [x for x in EXPERIMENT_ORDER if x in sample_map["FinalExperiment"].astype(str).unique()]

selected_experiment = st.sidebar.selectbox(
    "Experiment",
    options=["All"] + experiments,
)

plot_df = long_df.copy()
sample_map_plot = sample_map.copy()

if selected_experiment != "All":
    plot_df = plot_df[plot_df["FinalExperiment"].astype(str) == selected_experiment]
    sample_map_plot = sample_map_plot[sample_map_plot["FinalExperiment"].astype(str) == selected_experiment]

selected_metabolite = st.sidebar.selectbox("Metabolite", options=metabolites)

metadata_options = ["FinalGroup", "FinalExperiment", "Backgound", "Gender", "Date"]
group_by = st.sidebar.selectbox("Group x-axis", options=metadata_options, index=0)
color_by = st.sidebar.selectbox("Color by", options=metadata_options, index=0)
log_transform = st.sidebar.checkbox("Use log10(intensity + 1)", value=True)

# Ensure categorical ordering after filtering.
plot_df["FinalGroup"] = order_groups(plot_df["FinalGroup"].astype(str))
sample_map_plot["FinalGroup"] = order_groups(sample_map_plot["FinalGroup"].astype(str))


# -----------------------------
# Summary cards
# -----------------------------

n_metabolites = long_df["Metabolite"].nunique()
n_samples = sample_map["MetabolomicsSample"].nunique()
n_matched = int((sample_map["MatchStatus"] == "Matched").sum())
n_classes = long_df["MetaboliteClass"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Metabolites", n_metabolites)
col2.metric("Samples", n_samples)
col3.metric("Matched samples", n_matched)
col4.metric("Metabolite classes", n_classes)


# -----------------------------
# Tabs
# -----------------------------

tabs = st.tabs(
    [
        "Overview",
        "Metabolite explorer",
        "PCA",
        "Heatmap",
        "Volcano / differential",
        "Class explorer",
        "Correlation explorer",
        "Antibiotic recovery",
    ]
)

tab_overview, tab_explorer, tab_pca, tab_heatmap, tab_diff, tab_class, tab_corr, tab_recovery = tabs


with tab_overview:
    st.subheader("Detected sample mapping")
    st.caption("Samples are matched by mouse number and detected biological group. Group order is fixed biologically, especially for antibiotics.")

    st.dataframe(
        sample_map[
            [
                "MetabolomicsSample", "Mouse", "DetectedGroup", "FinalGroup", "FinalExperiment",
                "Gender", "Backgound", "Date", "MatchStatus", "Sample name",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Sample counts")
    count_df = (
        sample_map.groupby(["FinalExperiment", "FinalGroup"], observed=False)
        .size()
        .reset_index(name="n")
    )
    count_df = count_df[count_df["n"] > 0]

    fig_counts = px.bar(
        count_df,
        x="FinalGroup",
        y="n",
        color="FinalExperiment",
        text="n",
        title="Sample counts by group and experiment",
        category_orders={"FinalGroup": GROUP_ORDER, "FinalExperiment": EXPERIMENT_ORDER},
    )
    st.plotly_chart(fig_counts, use_container_width=True)

    st.subheader("Metabolite classes detected in this dataset")
    class_counts = (
        add_metabolite_classes(pd.DataFrame({"Metabolite": metabolites}))
        .groupby("MetaboliteClass", observed=False)
        .size()
        .reset_index(name="n")
    )
    class_counts = class_counts[class_counts["n"] > 0]
    fig_class_counts = px.bar(
        class_counts,
        x="n",
        y="MetaboliteClass",
        orientation="h",
        text="n",
        title="Rule-based metabolite classes from current metabolite names",
        category_orders={"MetaboliteClass": CLASS_ORDER},
    )
    fig_class_counts.update_layout(yaxis={"categoryorder": "array", "categoryarray": CLASS_ORDER[::-1]})
    st.plotly_chart(fig_class_counts, use_container_width=True)

    with st.expander("Show metabolites by class"):
        class_table = add_metabolite_classes(pd.DataFrame({"Metabolite": metabolites})).sort_values([
            "MetaboliteClass", "Metabolite"
        ])
        st.dataframe(class_table, use_container_width=True)

    st.subheader("Missingness")
    matrix_raw = make_matrix(met, log_transform=False)
    missing_by_metabolite = matrix_raw.isna().mean(axis=1).reset_index()
    missing_by_metabolite.columns = ["Metabolite", "MissingFraction"]
    missing_by_metabolite["MetaboliteClass"] = missing_by_metabolite["Metabolite"].map(classify_metabolite)

    fig_missing = px.histogram(
        missing_by_metabolite,
        x="MissingFraction",
        nbins=20,
        color="MetaboliteClass",
        title="Missing-value fraction across metabolites",
    )
    st.plotly_chart(fig_missing, use_container_width=True)


with tab_explorer:
    st.subheader(selected_metabolite)

    df_met = plot_df[plot_df["Metabolite"] == selected_metabolite].copy()
    df_met["DisplayIntensity"] = np.log10(df_met["Intensity"] + 1) if log_transform else df_met["Intensity"]

    fig_box = px.box(
        df_met,
        x=group_by,
        y="DisplayIntensity",
        color=color_by,
        points=False,
        hover_data=["MetabolomicsSample", "Mouse", "FinalGroup", "FinalExperiment", "Gender", "Backgound"],
        title=f"{selected_metabolite} — {classify_metabolite(selected_metabolite)}",
        category_orders={"FinalGroup": GROUP_ORDER},
    )
    fig_box = add_white_points_on_boxes(fig_box, df_met, group_by, "DisplayIntensity")
    fig_box.update_layout(xaxis_title=group_by, yaxis_title="Intensity")
    st.plotly_chart(fig_box, use_container_width=True)

    st.dataframe(
        df_met[["MetabolomicsSample", "Mouse", "FinalExperiment", "FinalGroup", "Gender", "Backgound", "Intensity"]],
        use_container_width=True,
    )


with tab_pca:
    st.subheader("PCA of metabolomics profiles")

    matrix = make_matrix(met, log_transform=log_transform)
    pca, pca_df, loadings = compute_pca(matrix, sample_map_plot, n_components=5)

    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_by,
        symbol="FinalExperiment",
        text="Mouse",
        hover_data=["MetabolomicsSample", "Mouse", "FinalGroup", "FinalExperiment", "Gender", "Backgound"],
        title="PCA",
        category_orders={"FinalGroup": GROUP_ORDER},
    )
    fig_pca.update_traces(textposition="top center")
    fig_pca.update_layout(xaxis_title=f"PC1 ({pc1:.1f}%)", yaxis_title=f"PC2 ({pc2:.1f}%)")
    st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Top PCA loadings")
    pc_choice = st.selectbox("Principal component for loadings", options=[c for c in loadings.columns if re.match(r"PC\d+", c)])
    load_table = loadings.copy()
    load_table["abs_loading"] = load_table[pc_choice].abs()
    load_table = load_table.sort_values("abs_loading", ascending=False)

    fig_loadings = px.bar(
        load_table.head(20).sort_values(pc_choice),
        x=pc_choice,
        y="Metabolite",
        color="MetaboliteClass",
        orientation="h",
        title=f"Top 20 absolute loadings for {pc_choice}",
    )
    st.plotly_chart(fig_loadings, use_container_width=True)
    st.dataframe(load_table[["Metabolite", "MetaboliteClass", pc_choice, "abs_loading"]].head(50), use_container_width=True)


with tab_heatmap:
    st.subheader("Top variable metabolites")

    n_top = st.slider("Number of metabolites", min_value=10, max_value=150, value=50, step=10)
    class_filter_heat = st.multiselect(
        "Optional metabolite class filter",
        options=[c for c in CLASS_ORDER if c in long_df["MetaboliteClass"].astype(str).unique()],
        default=[],
    )

    matrix = make_matrix(met, log_transform=log_transform)
    selected_samples = ordered_samples(sample_map_plot)
    matrix = matrix[selected_samples]

    candidate_metabolites = matrix.index.tolist()
    if class_filter_heat:
        candidate_metabolites = [m for m in candidate_metabolites if classify_metabolite(m) in class_filter_heat]

    variances = matrix.loc[candidate_metabolites].var(axis=1).sort_values(ascending=False)
    top_metabolites = variances.head(n_top).index

    heat = matrix.loc[top_metabolites]
    heat_z = heat.sub(heat.mean(axis=1), axis=0).div(heat.std(axis=1), axis=0)
    heat_z = heat_z.replace([np.inf, -np.inf], np.nan).fillna(0)

    fig_heat = px.imshow(
        heat_z[selected_samples],
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"Top {min(n_top, len(top_metabolites))} variable metabolites, row z-score",
    )
    fig_heat.update_layout(height=900)
    st.plotly_chart(fig_heat, use_container_width=True)


with tab_diff:
    st.subheader("Volcano / differential analysis")

    comparisons = {
        "GF vs SPF": ("GF", "SPF"),
        "SPF Before Abx vs SPF Abx": ("SPF Before Abx", "SPF Abx"),
        "SPF Abx vs SPF After Abx": ("SPF Abx", "SPF After Abx"),
        "SPF Before Abx vs SPF After Abx": ("SPF Before Abx", "SPF After Abx"),
    }

    available_groups = ordered_unique(plot_df["FinalGroup"].astype(str))
    valid_comparisons = {
        k: v for k, v in comparisons.items() if v[0] in available_groups and v[1] in available_groups
    }
    comparison_mode = st.radio("Comparison mode", options=["Preset", "Custom"], horizontal=True)

    if comparison_mode == "Preset" and valid_comparisons:
        comparison_name = st.selectbox("Preset comparison", options=list(valid_comparisons.keys()))
        group_a, group_b = valid_comparisons[comparison_name]
    else:
        col_a, col_b = st.columns(2)
        group_a = col_a.selectbox("Reference group", options=available_groups, index=0)
        group_b = col_b.selectbox("Comparison group", options=available_groups, index=min(1, len(available_groups) - 1))

    if group_a == group_b:
        st.warning("Choose two different groups.")
    else:
        diff_df = differential_test(plot_df, group_col="FinalGroup", group_a=group_a, group_b=group_b, log_transform=log_transform)
        effect_col = "difference_log10_mean" if log_transform else "difference_mean"

        st.caption(
            "Effect size is comparison minus reference on the selected scale. Test: Welch's t-test; multiple testing: Benjamini-Hochberg FDR."
        )

        fdr_cutoff = st.slider("FDR cutoff", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
        effect_cutoff = st.slider("Absolute effect-size cutoff", min_value=0.0, max_value=2.0, value=0.25, step=0.05)

        volcano = diff_df.dropna(subset=[effect_col, "minus_log10_p"]).copy()
        volcano["Significant"] = (volcano["FDR"] < fdr_cutoff) & (volcano[effect_col].abs() >= effect_cutoff)

        fig_volcano = px.scatter(
            volcano,
            x=effect_col,
            y="minus_log10_p",
            color="MetaboliteClass",
            symbol="Significant",
            hover_data=["Metabolite", "p_value", "FDR", "log2FC_raw_mean_with_pseudocount1", "n_reference", "n_comparison"],
            title=f"{group_b} vs {group_a}",
        )
        fig_volcano.add_vline(x=effect_cutoff, line_dash="dash")
        fig_volcano.add_vline(x=-effect_cutoff, line_dash="dash")
        fig_volcano.update_layout(xaxis_title=f"{group_b} minus {group_a}", yaxis_title="-log10(p-value)")
        st.plotly_chart(fig_volcano, use_container_width=True)

        col_top1, col_top2 = st.columns(2)
        with col_top1:
            st.subheader("Top increased")
            st.dataframe(diff_df.sort_values(effect_col, ascending=False).head(15), use_container_width=True)
        with col_top2:
            st.subheader("Top decreased")
            st.dataframe(diff_df.sort_values(effect_col, ascending=True).head(15), use_container_width=True)

        st.subheader("Differential table")
        st.dataframe(diff_df, use_container_width=True)

        csv = diff_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download differential table",
            data=csv,
            file_name=f"differential_{group_b}_vs_{group_a}.csv".replace(" ", "_"),
            mime="text/csv",
        )


with tab_class:
    st.subheader("Class explorer")
    st.caption("Classes are rule-based from the metabolite names present in this exact dataset.")

    class_options = [c for c in CLASS_ORDER if c in long_df["MetaboliteClass"].astype(str).unique()]
    selected_class = st.selectbox("Metabolite class", options=class_options)

    class_mets = sorted(long_df.loc[long_df["MetaboliteClass"].astype(str) == selected_class, "Metabolite"].unique())
    st.write(f"Detected metabolites in class: {len(class_mets)}")
    st.dataframe(pd.DataFrame({"Metabolite": class_mets}), use_container_width=True)

    class_df = plot_df[plot_df["Metabolite"].isin(class_mets)].copy()
    class_df["DisplayIntensity"] = np.log10(class_df["Intensity"] + 1) if log_transform else class_df["Intensity"]

    group_summary = (
        class_df.groupby(["FinalGroup", "Metabolite"], observed=False)["DisplayIntensity"]
        .mean()
        .reset_index()
    )
    group_summary = group_summary.dropna(subset=["DisplayIntensity"])

    fig_class = px.box(
        group_summary,
        x="FinalGroup",
        y="DisplayIntensity",
        points=False,
        title=f"Class-level distribution of group mean intensities: {selected_class}",
        category_orders={"FinalGroup": GROUP_ORDER},
    )
    fig_class = add_white_points_on_boxes(fig_class, group_summary, "FinalGroup", "DisplayIntensity", sample_col="Metabolite")
    st.plotly_chart(fig_class, use_container_width=True)

    matrix = make_matrix(met, log_transform=log_transform)
    selected_samples = ordered_samples(sample_map_plot)
    matrix = matrix.loc[class_mets, selected_samples]
    heat_z = matrix.sub(matrix.mean(axis=1), axis=0).div(matrix.std(axis=1), axis=0)
    heat_z = heat_z.replace([np.inf, -np.inf], np.nan).fillna(0)
    fig_class_heat = px.imshow(
        heat_z,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"{selected_class}: row z-score heatmap",
    )
    fig_class_heat.update_layout(height=max(450, 22 * len(class_mets)))
    st.plotly_chart(fig_class_heat, use_container_width=True)


with tab_corr:
    st.subheader("Correlation explorer")
    st.caption("Spearman correlations across the currently selected experiment/samples.")

    matrix = make_matrix(met, log_transform=log_transform)
    selected_samples = ordered_samples(sample_map_plot)
    matrix = matrix[selected_samples]

    corr_met = st.selectbox("Anchor metabolite", options=metabolites, index=metabolites.index(selected_metabolite))
    corr_df = spearman_correlations(matrix, corr_met)

    n_show = st.slider("Number of top correlated metabolites", min_value=5, max_value=50, value=20, step=5)
    st.dataframe(corr_df.head(n_show), use_container_width=True)

    fig_corr_bar = px.bar(
        corr_df.head(n_show).sort_values("spearman_rho"),
        x="spearman_rho",
        y="Metabolite",
        color="MetaboliteClass",
        orientation="h",
        title=f"Top correlations with {corr_met}",
    )
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    scatter_target = st.selectbox("Scatter target metabolite", options=corr_df["Metabolite"].dropna().tolist())
    scatter_df = pd.DataFrame({
        corr_met: matrix.loc[corr_met],
        scatter_target: matrix.loc[scatter_target],
    }).reset_index().rename(columns={"index": "MetabolomicsSample"})
    scatter_df = scatter_df.merge(sample_map, on="MetabolomicsSample", how="left")

    rho_row = corr_df[corr_df["Metabolite"] == scatter_target].iloc[0]
    fig_scatter = px.scatter(
        scatter_df,
        x=corr_met,
        y=scatter_target,
        color="FinalGroup",
        symbol="FinalExperiment",
        trendline="ols",
        hover_data=["MetabolomicsSample", "Mouse", "FinalGroup", "FinalExperiment"],
        title=f"{corr_met} vs {scatter_target}: Spearman rho={rho_row['spearman_rho']:.2f}, p={rho_row['p_value']:.3g}",
        category_orders={"FinalGroup": GROUP_ORDER},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


with tab_recovery:
    st.subheader("Antibiotic recovery")
    st.caption("Order is fixed as SPF Before Abx → SPF Abx → SPF After Abx.")

    abx_df = long_df[long_df["FinalGroup"].astype(str).isin(ABX_ORDER)].copy()
    abx_df["FinalGroup"] = pd.Categorical(abx_df["FinalGroup"].astype(str), categories=ABX_ORDER, ordered=True)
    abx_df["DisplayIntensity"] = np.log10(abx_df["Intensity"] + 1) if log_transform else abx_df["Intensity"]

    recovery_df = recovery_table(long_df, log_transform=log_transform)

    st.subheader("Recovery score table")
    st.caption("RecoveryScore = (After - Abx) / (Before - Abx). Around 1 suggests recovery toward the pre-antibiotic state; around 0 suggests no recovery; >1 suggests overshoot. Interpret cautiously when Before - Abx is close to zero.")
    st.dataframe(recovery_df, use_container_width=True)

    recovery_metabolite = st.selectbox("Metabolite trajectory", options=metabolites, index=metabolites.index(selected_metabolite))
    traj = abx_df[abx_df["Metabolite"] == recovery_metabolite].copy()

    fig_traj = px.line(
        traj.sort_values(["Mouse", "FinalGroup"]),
        x="FinalGroup",
        y="DisplayIntensity",
        color="Mouse",
        markers=True,
        hover_data=["MetabolomicsSample", "Mouse", "FinalGroup", "Gender", "Backgound"],
        title=f"Antibiotic trajectory: {recovery_metabolite}",
        category_orders={"FinalGroup": ABX_ORDER},
    )
    fig_traj.update_layout(xaxis_title="Antibiotic phase", yaxis_title="Intensity")
    st.plotly_chart(fig_traj, use_container_width=True)

    top_recovery = recovery_df.dropna(subset=["RecoveryScore"]).copy()
    top_recovery = top_recovery[np.isfinite(top_recovery["RecoveryScore"])]
    top_recovery["abs_deviation_from_1"] = (top_recovery["RecoveryScore"] - 1).abs()

    fig_recovery = px.scatter(
        top_recovery,
        x="Before_minus_Abx",
        y="After_minus_Abx",
        color="MetaboliteClass",
        hover_data=["Metabolite", "RecoveryScore"],
        title="Antibiotic response and recovery across metabolites",
    )
    fig_recovery.add_shape(type="line", x0=top_recovery["Before_minus_Abx"].min(), x1=top_recovery["Before_minus_Abx"].max(), y0=top_recovery["Before_minus_Abx"].min(), y1=top_recovery["Before_minus_Abx"].max(), line=dict(dash="dash"))
    fig_recovery.update_layout(xaxis_title="Before - Abx", yaxis_title="After - Abx")
    st.plotly_chart(fig_recovery, use_container_width=True)

    csv = recovery_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recovery table",
        data=csv,
        file_name="antibiotic_recovery_table.csv",
        mime="text/csv",
    )
