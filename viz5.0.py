import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import tempfile
import re
import os

# ────────────────────────────────────────────────
# Page config + styling
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Insights • Smart Data Explorer",
    page_icon="🧠📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom styling
st.markdown("""
    <style>
    .main .block-container {padding-top: 1.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(0,0,0,0.05);
        border-radius: 6px 6px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    h1, h2, h3 {color: #1e40af;}
    .stButton>button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Insights")
st.caption("Upload data • Ask natural language questions • Visualize • Get insights")

# ────────────────────────────────────────────────
# Sidebar – only upload + quick actions
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Input")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"],
        help="CSV or .xlsx files supported"
    )

    if uploaded_file:
        st.success("File uploaded successfully")

    st.divider()

    st.subheader("Quick Actions")
    show_preview = st.checkbox("Show data preview", value=True)
    enable_cleaning = st.checkbox("Basic cleaning options", value=False)

# ────────────────────────────────────────────────
# Session state for data persistence
# ────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None
if 'query_response' not in st.session_state:
    st.session_state.query_response = None

# ────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Optional basic cleaning
        if enable_cleaning:
            df = df.drop_duplicates()
            df = df.dropna(how='all')  # only drop fully empty rows

        st.session_state.df = df

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

# ────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    # ───── Tabs ─────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data & Preview",
        "🗣️ Ask Questions",
        "🔍 AI Insights & Report",
        "📈 Visualizations"
    ])

    # ───── Tab 1: Data Preview ─────
    with tab1:
        st.subheader("Data Overview")

        col_left, col_right = st.columns([3, 1])

        with col_left:
            if show_preview:
                with st.expander("First 1,000 rows (scrollable)", expanded=True):
                    st.dataframe(df.head(1000), use_container_width=True)

        with col_right:
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", df.shape[1])
            st.metric("Missing cells", df.isnull().sum().sum())

        st.subheader("Column Types")
        dtypes = df.dtypes.value_counts().to_frame(name="Count")
        st.bar_chart(dtypes)

    # ───── Tab 2: Natural Language Query ─────
    with tab2:
        st.subheader("Ask a question about your data")

        query = st.text_input(
            "Example: What is the average sales per region?",
            placeholder="Type your question here...",
            key="query_input"
        )

        if st.button("🔎 Ask", type="primary", use_container_width=True) and query:
            with st.spinner("Processing your question..."):
                # ───── Your original query processor ─────
                def process_query(q):
                    q = q.lower()
                    if "average" in q and "per" in q:
                        match = re.search(r"average.*of.*(\w+).*per.*(\w+)", q)
                        if match:
                            col_val, col_group = match.groups()
                            if col_val in df.columns and col_group in df.columns:
                                result = df.groupby(col_group)[col_val].mean()
                                return f"**Average {col_val} per {col_group}:**\n\n{result.to_string()}"
                            else:
                                return "One or both columns not found in the dataset."
                        else:
                            return "Format suggestion: 'average of sales per region'"

                    elif "sum" in q and "per" in q:
                        match = re.search(r"sum.*of.*(\w+).*per.*(\w+)", q)
                        if match:
                            col_val, col_group = match.groups()
                            if col_val in df.columns and col_group in df.columns:
                                result = df.groupby(col_group)[col_val].sum()
                                return f"**Total {col_val} per {col_group}:**\n\n{result.to_string()}"
                            else:
                                return "One or both columns not found."
                        else:
                            return "Format suggestion: 'sum of revenue per country'"

                    return ("Sorry, I couldn't understand the question.\n\n"
                            "Currently supported patterns:\n"
                            "• average of [column] per [column]\n"
                            "• sum of [column] per [column]")

                response = process_query(query)
                st.session_state.query_response = response

        # Show result in nice format
        if st.session_state.query_response:
            with st.container(border=True):
                st.markdown("**Your question:** " + query)
                st.divider()
                st.markdown(st.session_state.query_response)

    # ───── Tab 3: AI Insights & Profiling ─────
    with tab3:
        st.subheader("AI-Powered Insights & Profiling")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate Full Profiling Report", type="primary"):
                with st.spinner("Generating pandas-profiling report (this may take 10–90 seconds)..."):
                    pr = ProfileReport(df, explorative=True, title="Data Profiling Report")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                        pr.to_file(tmpfile.name)
                        tmp_path = tmpfile.name

                    with open(tmp_path, "r", encoding="utf-8") as f:
                        html_content = f.read()

                    st.components.v1.html(html_content, height=1000, scrolling=True)

                    # Cleanup
                    os.unlink(tmp_path)

        with col2:
            missing = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()

            st.metric("Missing Values (total)", missing)
            st.metric("Duplicate Rows", duplicates)

            summary_text = f"🔹 Missing Values: {missing:,}\n🔹 Duplicated Rows: {duplicates:,}"

            st.download_button(
                label="📥 Download Quick Summary",
                data=summary_text,
                file_name="data_summary.txt",
                mime="text/plain"
            )

    # ───── Tab 4: Comparative Visualizations ─────
    with tab4:
        st.subheader("Custom Comparative Visualizations")

        chart_type = st.selectbox(
            "Chart Type",
            ["Scatter", "Bar", "Line", "Heatmap"],
            index=0
        )

        col_left, col_right = st.columns(2)

        with col_left:
            x_axes = st.multiselect(
                "X-axis variable(s)",
                options=df.columns.tolist(),
                default=[df.columns[0]] if len(df.columns) > 0 else []
            )

        with col_right:
            y_axes = st.multiselect(
                "Y-axis variable(s)",
                options=df.columns.tolist(),
                default=[df.columns[1]] if len(df.columns) > 1 else []
            )

        if st.button("Generate Plots", type="primary"):
            if chart_type == "Heatmap":
                if df.select_dtypes(include="number").shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        df.corr(numeric_only=True),
                        annot=True,
                        cmap="coolwarm",
                        fmt=".2f",
                        ax=ax
                    )
                    st.pyplot(fig)
                else:
                    st.warning("Not enough numeric columns for correlation heatmap.")
            else:
                if not x_axes or not y_axes:
                    st.warning("Please select at least one X and one Y variable.")
                else:
                    for x in x_axes:
                        for y in y_axes:
                            if x == y:
                                continue
                            try:
                                if chart_type == "Scatter":
                                    fig = px.scatter(df, x=x, y=y, title=f"Scatter: {y} vs {x}")
                                elif chart_type == "Bar":
                                    fig = px.bar(df, x=x, y=y, title=f"Bar: {y} by {x}")
                                elif chart_type == "Line":
                                    fig = px.line(df.sort_values(x), x=x, y=y, title=f"Line: {y} over {x}")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not create plot for {x} vs {y}: {str(e)}")

else:
    # Welcome screen when no file is uploaded
    st.info("Upload a CSV or Excel file to start exploring your data.", icon="⬆️")
    st.markdown("""
    ### Supported features:
    • Natural language questions (average / sum patterns)  
    • Interactive pandas-profiling report  
    • Multi-variable scatter/bar/line charts  
    • Correlation heatmap  
    • Data preview & basic metrics
    """)

st.markdown("---")
st.caption("AI Insights v2 • Built with Streamlit • 2026")
