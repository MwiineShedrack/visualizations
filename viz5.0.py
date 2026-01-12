import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import tempfile
import os
import numpy as np

# ────────────────────────────────────────────────
# Page config + improved styling
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Insights • Smart Data Explorer",
    page_icon="🧠📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 12px; justify-content: center;}
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        white-space: pre-wrap;
        background-color: rgba(0,0,0,0.04);
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }
    h1, h2, h3 {color: #1e40af;}
    .stButton>button {width: 100%; border-radius: 6px;}
    hr {margin: 1.5rem 0;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Insights")
st.caption("Upload your data → discover hidden patterns & insights automatically")

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx"],
        help="Supported formats: .csv, .xlsx"
    )

    if uploaded_file:
        st.success("File loaded")

    st.divider()
    st.subheader("Display Options")
    show_preview = st.checkbox("Show data preview", value=True)
    basic_clean = st.checkbox("Remove duplicates & fully-empty rows", value=True)

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None

# ────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if basic_clean:
            df = df.drop_duplicates()
            df = df.dropna(how='all')

        st.session_state.df = df

    except Exception as e:
        st.error(f"Could not read file: {str(e)}")
        st.stop()

# ────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    tab_preview, tab_insights, tab_report, tab_viz = st.tabs([
        "📋 Data & Preview",
        "🔍 Key Insights",
        "📊 Profiling Report",
        "📈 Visualizations"
    ])

    # ───── Tab 1: Data Preview ─────
    with tab_preview:
        st.subheader("Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())

        if show_preview:
            with st.expander("Data Preview (first 800 rows)", expanded=True):
                st.dataframe(df.head(800), use_container_width=True)

        with st.expander("Column Information"):
            col_info = pd.DataFrame({
                "Type": df.dtypes,
                "Unique": df.nunique(),
                "% Missing": (df.isnull().mean() * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

    # ───── Tab 2: Functional / Time-consuming Insights ─────
    with tab_insights:
        st.subheader("Automatically Detected Key Insights")
        st.caption("These are patterns that usually take analysts significant time to uncover manually")

        # 1. Strongest correlations
        with st.expander("Strongest Linear Relationships (Correlation > |0.7|)", expanded=True):
            numeric = df.select_dtypes(include=np.number)
            if numeric.shape[1] >= 2:
                corr = numeric.corr().abs()
                corr_triu = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                strong = corr_triu.stack().reset_index()
                strong.columns = ['Var1', 'Var2', 'Corr']
                strong = strong.sort_values('Corr', ascending=False).query('Corr > 0.7')

                if not strong.empty:
                    st.dataframe(strong.style.format({'Corr': '{:.3f}'}).background_gradient(cmap='YlOrRd', subset=['Corr']))
                    st.caption("High values may indicate multicollinearity or redundant features.")
                else:
                    st.info("No correlations stronger than |0.7| found.")
            else:
                st.info("Not enough numeric columns.")

        # 2. Outliers summary
        with st.expander("Outlier Summary (IQR method)"):
            outliers = {}
            for col in df.select_dtypes(include=np.number).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                count = ((df[col] < lb) | (df[col] > ub)).sum()
                if count > 0:
                    outliers[col] = count
            if outliers:
                out_df = pd.DataFrame.from_dict(outliers, orient='index', columns=['Outlier Count'])
                out_df['% of rows'] = (out_df['Outlier Count'] / len(df) * 100).round(2)
                st.dataframe(out_df.sort_values('Outlier Count', ascending=False))
            else:
                st.success("No significant outliers detected (IQR method).")

        # 3. Highly skewed columns
        with st.expander("Highly Skewed Numeric Columns (skew > |2| or < -2)"):
            skews = df.select_dtypes(include=np.number).skew().dropna()
            high_skew = skews[abs(skews) > 2].sort_values(key=abs, ascending=False)
            if not high_skew.empty:
                st.dataframe(high_skew.to_frame(name='Skewness').style.background_gradient(cmap='OrRd', subset=['Skewness']))
                st.caption("High skew often means log/power transformation or non-parametric methods may help.")
            else:
                st.info("No strongly skewed numeric columns detected.")

        # 4. Categorical imbalance
        with st.expander("Highly Imbalanced Categories"):
            imbal = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                vc = df[col].value_counts(normalize=True)
                if len(vc) > 1 and vc.iloc[0] > 0.6:  # top category >60%
                    imbal[col] = f"Top category = {vc.index[0]} ({vc.iloc[0]:.1%})"
            if imbal:
                st.write(pd.Series(imbal).to_frame(name="Imbalance Note"))
            else:
                st.info("No strongly imbalanced categorical columns (>60% in one category).")

        # 5. Missing value hotspots
        with st.expander("Columns with High Missingness (>30%)"):
            miss = (df.isnull().mean() * 100).round(2)
            high_miss = miss[miss > 30].sort_values(ascending=False)
            if not high_miss.empty:
                st.dataframe(high_miss.to_frame(name='% Missing').style.background_gradient(cmap='Reds'))
            else:
                st.success("No columns with >30% missing values.")

    # ───── Tab 3: Profiling Report ─────
    with tab_report:
        st.subheader("Full Automated Profiling Report")
        st.caption("Detailed EDA — may take 10–90 seconds depending on dataset size")

        if st.button("Generate & View Report", type="primary"):
            with st.spinner("Creating pandas-profiling report..."):
                pr = ProfileReport(df, explorative=True, title="Data Profile Report")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    pr.to_file(tmp.name)
                    tmp_path = tmp.name

                with open(tmp_path, "r", encoding="utf-8") as f:
                    html = f.read()

                st.components.v1.html(html, height=1000, scrolling=True)
                os.unlink(tmp_path)

        # Quick summary download
        missing_total = df.isnull().sum().sum()
        dup_rows = df.duplicated().sum()
        quick_txt = f"Missing cells: {missing_total:,}\nDuplicate rows: {dup_rows:,}"
        st.download_button("📥 Download Quick Summary", quick_txt, "quick_summary.txt", "text/plain")

    # ───── Tab 4: Visualizations ─────
    with tab_viz:
        st.subheader("Comparative Visualizations")

        chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Line", "Heatmap"])

        colA, colB = st.columns(2)
        with colA:
            xs = st.multiselect("X variable(s)", df.columns.tolist())
        with colB:
            ys = st.multiselect("Y variable(s)", df.columns.tolist())

        if st.button("Generate Charts", type="primary"):
            if chart_type == "Heatmap":
                numeric_df = df.select_dtypes(include=np.number)
                if numeric_df.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 2 numeric columns.")
            else:
                if not xs or not ys:
                    st.warning("Select at least one X and one Y variable.")
                else:
                    for x in xs:
                        for y in ys:
                            if x == y: continue
                            try:
                                if chart_type == "Scatter":
                                    fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
                                elif chart_type == "Bar":
                                    fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
                                elif chart_type == "Line":
                                    fig = px.line(df.sort_values(x), x=x, y=y, title=f"{y} over {x}")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as ex:
                                st.error(f"Plot failed for {x} vs {y}: {ex}")

else:
    st.info("Upload a CSV or Excel file to unlock insights and visualizations.", icon="⬆️")

st.markdown("---")
st.caption("AI Insights • Enhanced Edition • Nairobi, 2026")
