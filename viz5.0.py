import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────
# Page config & styling
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
st.caption("Upload your data → get smart summaries, insights & visualizations")

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

    tab_preview, tab_insights, tab_summary, tab_viz = st.tabs([
        "📋 Data Preview",
        "🔍 Key Insights",
        "📊 Data Summary",
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

    # ───── Tab 2: Key Insights ─────
    with tab_insights:
        st.subheader("Automatically Detected Key Insights")
        st.caption("Patterns that usually take significant manual analysis time")

        # Strong correlations
        with st.expander("Strongest Correlations (|r| > 0.7)", expanded=True):
            numeric = df.select_dtypes(include=np.number)
            if numeric.shape[1] >= 2:
                corr = numeric.corr().abs()
                corr_triu = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                strong = corr_triu.stack().reset_index()
                strong.columns = ['Var1', 'Var2', 'Corr']
                strong = strong.sort_values('Corr', ascending=False).query('Corr > 0.7')
                if not strong.empty:
                    st.dataframe(strong.style.format({'Corr': '{:.3f}'})
                                 .background_gradient(cmap='YlOrRd', subset=['Corr']))
                else:
                    st.info("No correlations stronger than |0.7| found.")
            else:
                st.info("Not enough numeric columns.")

        # Outliers
        with st.expander("Outlier Summary (IQR method)"):
            outliers = {}
            for col in numeric.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if count > 0:
                    outliers[col] = count
            if outliers:
                out_df = pd.DataFrame.from_dict(outliers, orient='index', columns=['Outlier Count'])
                out_df['% of rows'] = (out_df['Outlier Count'] / len(df) * 100).round(2)
                st.dataframe(out_df.sort_values('Outlier Count', ascending=False))
            else:
                st.success("No significant outliers detected.")

        # Skewed columns
        with st.expander("Highly Skewed Numeric Columns (|skew| > 2)"):
            skews = numeric.skew().dropna()
            high_skew = skews[abs(skews) > 2].sort_values(key=abs, ascending=False)
            if not high_skew.empty:
                st.dataframe(high_skew.to_frame(name='Skewness')
                             .style.background_gradient(cmap='OrRd', subset=['Skewness']))
            else:
                st.info("No strongly skewed numeric columns.")

        # Imbalanced categoricals
        with st.expander("Highly Imbalanced Categorical Columns"):
            imbal = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                vc = df[col].value_counts(normalize=True)
                if len(vc) > 1 and vc.iloc[0] > 0.6:
                    imbal[col] = f"Top: {vc.index[0]} ({vc.iloc[0]:.1%})"
            if imbal:
                st.write(pd.Series(imbal).to_frame(name="Imbalance Note"))
            else:
                st.info("No strongly imbalanced categories (>60% in one value).")

        # High missing columns
        with st.expander("Columns with >30% Missing Values"):
            miss = (df.isnull().mean() * 100).round(2)
            high_miss = miss[miss > 30].sort_values(ascending=False)
            if not high_miss.empty:
                st.dataframe(high_miss.to_frame(name='% Missing')
                             .style.background_gradient(cmap='Reds'))
            else:
                st.success("No columns with >30% missing.")

    # ───── Tab 3: Data Summary (fixed styler) ─────
    with tab_summary:
        st.subheader("Quick Data Summary")

        # Descriptive statistics - safe formatting only on numeric columns
        st.markdown("**Descriptive Statistics**")
        desc = df.describe(include="all").T
        # Create a formatter dict: apply {:.2f} only to numeric summary rows
        numeric_rows = desc.index.intersection(['mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        formatter = {col: "{:.2f}" if row in numeric_rows else "{}" for row, col in desc.index.to_flat_index()}
        st.dataframe(desc.style.format(formatter))

        # Missing values
        st.markdown("**Missing Values**")
        miss = df.isnull().sum()
        miss_pct = (miss / len(df) * 100).round(2)
        miss_df = pd.DataFrame({"Missing": miss, "% Missing": miss_pct}) \
                  .sort_values("Missing", ascending=False)
        st.dataframe(miss_df.style.format({"% Missing": "{:.2f}%"}))

        # Column types & examples
        st.markdown("**Column Types & Examples**")
        overview = pd.DataFrame({
            "Type": df.dtypes,
            "Unique Values": df.nunique(),
            "Example": [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
        })
        st.dataframe(overview)

    # ───── Tab 4: Visualizations ─────
    with tab_viz:
        st.subheader("Custom Visualizations")

        chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Line", "Box", "Histogram", "Heatmap"])

        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X axis", df.columns, key="x_viz")
        with col2:
            y_var = st.selectbox("Y axis / Category", df.columns, index=1 if len(df.columns)>1 else 0, key="y_viz")

        if st.button("Generate Visualization", type="primary"):
            try:
                if chart_type == "Heatmap":
                    num_df = df.select_dtypes(include=np.number)
                    if num_df.shape[1] >= 2:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("Need at least 2 numeric columns.")
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_var, y=y_var)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_var, y=y_var)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "Line":
                    fig = px.line(df.sort_values(x_var), x=x_var, y=y_var)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "Box":
                    fig = px.box(df, x=x_var, y=y_var)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=y_var)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization failed: {str(e)}")

else:
    st.info("Upload a CSV or Excel file to start exploring your data.", icon="⬆️")

st.markdown("---")
st.caption("AI Insights • Lightweight & Cloud-friendly • Nairobi, January 2026")
