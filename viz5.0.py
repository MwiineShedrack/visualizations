import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport  # Updated from pandas_profiling
import tempfile
import os
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv

load_dotenv()  # optional — loads .env file if present

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Insights • Smart Data Explorer",
    page_icon="🧠📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Custom CSS for better look
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 1px solid #ddd; }
    h1, h2, h3 { color: #1e3a8a; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 6px; }
    .stTextInput>div>div>input { border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Title & description
# ────────────────────────────────────────────────
st.title("🧠 AI Insights – Talk to Your Data")
st.markdown("Upload your dataset and ask anything in natural language — averages, trends, plots, comparisons, forecasts, and more.")

# ────────────────────────────────────────────────
# Sidebar – Controls
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data Controls")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], help="Max ~50–100 MB depending on hosting limits")

    st.divider()
    st.subheader("🤖 AI Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    llm_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

    st.divider()
    st.subheader("🛠 Quick Cleaning")
    clean_missing = st.checkbox("Drop rows with missing values", value=False)
    remove_duplicates = st.checkbox("Remove duplicate rows", value=False)

# ────────────────────────────────────────────────
# Session state initialization
# ────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sdf" not in st.session_state:
    st.session_state.sdf = None

# ────────────────────────────────────────────────
# Load & Prepare Data
# ────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df

        # Apply quick cleaning if requested
        clean_df = df.copy()
        if remove_duplicates:
            clean_df = clean_df.drop_duplicates()
        if clean_missing:
            clean_df = clean_df.dropna()

        st.session_state.clean_df = clean_df

        # Initialize PandasAI SmartDataframe
        if api_key:
            try:
                llm = OpenAI(api_token=api_key, model=llm_model)
                st.session_state.sdf = SmartDataframe(
                    clean_df,
                    config={
                        "llm": llm,
                        "enable_cache": False,          # avoid stale answers during dev
                        "save_charts": True,
                        "save_charts_path": "exports/charts",
                        "verbose": True
                    }
                )
                st.sidebar.success("AI is ready! Ask anything.")
            except Exception as e:
                st.sidebar.error(f"Could not initialize PandasAI: {e}")
        else:
            st.sidebar.warning("Add OpenAI API key to unlock natural language queries.")

        # Show preview
        st.subheader("Data Preview")
        st.dataframe(clean_df.head(8), use_container_width=True)

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state.df = None

# ────────────────────────────────────────────────
# Tabs for better organization
# ────────────────────────────────────────────────
if st.session_state.df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat with Data", "📈 Visualizations", "🔍 Auto Insights", "🛠 Data Cleaning"])

    with tab1:
        st.subheader("Ask anything about your data")

        # Display chat history
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        # Chat input
        if prompt := st.chat_input("What would you like to know? (e.g. 'top 5 products by revenue', 'plot sales trend by month', 'correlation between age and income')"):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.sdf is None:
                        st.warning("Please provide OpenAI API key first.")
                    else:
                        try:
                            response = st.session_state.sdf.chat(prompt)

                            # PandasAI can return str, pd.DataFrame, plot path, etc.
                            if isinstance(response, pd.DataFrame):
                                st.dataframe(response, use_container_width=True)
                                st.session_state.chat_history.append(("assistant", "Here's the resulting table:"))
                            elif isinstance(response, str) and "exports/charts" in response:
                                # Show generated chart if path returned
                                if os.path.exists(response):
                                    st.image(response, caption="Generated Visualization")
                                st.markdown("**Generated plot saved.**")
                                st.session_state.chat_history.append(("assistant", "I created a chart for you!"))
                            else:
                                st.markdown(response)
                                st.session_state.chat_history.append(("assistant", response))

                        except Exception as e:
                            error_msg = f"Sorry, I couldn't process that. Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append(("assistant", error_msg))

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with tab2:
        st.subheader("Quick Comparative Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            x = st.selectbox("X axis", st.session_state.clean_df.columns, key="viz_x")
        with col2:
            y = st.selectbox("Y axis", st.session_state.clean_df.columns, key="viz_y")

        chart_type = st.selectbox("Chart type", ["Scatter", "Line", "Bar", "Area", "Box"])

        if x and y:
            if chart_type == "Scatter":
                fig = px.scatter(st.session_state.clean_df, x=x, y=y, title=f"{y} vs {x}")
            elif chart_type == "Line":
                fig = px.line(st.session_state.clean_df, x=x, y=y, title=f"{y} over {x}")
            elif chart_type == "Bar":
                fig = px.bar(st.session_state.clean_df, x=x, y=y, title=f"{y} by {x}")
            elif chart_type == "Area":
                fig = px.area(st.session_state.clean_df, x=x, y=y, title=f"{y} area over {x}")
            elif chart_type == "Box":
                fig = px.box(st.session_state.clean_df, x=x, y=y, title=f"Box plot: {y} by {x}")

            st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        if st.checkbox("Show Correlation Heatmap"):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(st.session_state.clean_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    with tab3:
        st.subheader("Automatic Exploratory Analysis (ydata-profiling)")
        if st.button("Generate Full Report (may take 10–60s)"):
            with st.spinner("Generating profiling report..."):
                profile = ProfileReport(
                    st.session_state.clean_df,
                    title="Data Profiling Report",
                    explorative=True,
                    minimal=False
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    profile.to_file(tmp_file.name)
                    tmp_file_path = tmp_file.name

                with open(tmp_file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                st.components.v1.html(html_content, height=1000, scrolling=True)

                # Clean up
                os.unlink(tmp_file_path)

    with tab4:
        st.subheader("Data Cleaning & Export")
        st.write("Current shape:", st.session_state.clean_df.shape)

        if st.button("Download Cleaned CSV"):
            csv = st.session_state.clean_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download cleaned_data.csv",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

else:
    st.info("↑ Upload your CSV or Excel file to begin.", icon="ℹ️")
