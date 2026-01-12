import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import tempfile
import os

# Try to use Ollama → fall back to smart rule-based if not available
try:
    from pandasai import SmartDataframe
    from pandasai.llm import Ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

# ─────────────────────────────────────
st.set_page_config(page_title="AI Insights • Free Forever", layout="wide", page_icon="🧠")
st.title("🧠 AI Insights – 100% Free & Offline")
st.markdown("**No API keys • No costs • Works offline after one-time setup**")

# Custom CSS
st.markdown("<style>.css-1d391kg {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("CSV or Excel", type=["csv", "xlsx"])

    st.divider()
    st.header("🛠 Quick Clean")
    drop_na = st.checkbox("Drop rows with missing values")
    drop_dup = st.checkbox("Remove duplicates")

    st.divider()
    if OLLAMA_AVAILABLE:
        model_name = st.selectbox("Local AI Model", ["phi3:mini", "llama3.2", "gemma2:2b"], index=0)
    else:
        st.warning("Ollama not detected → using smart rule-based AI (still powerful!)")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.chat = []

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    
    if drop_dup:
        df = df.drop_duplicates()
    if drop_na:
        df = df.dropna()
    
    st.session_state.df = df
    st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

if st.session_state.df is not None:
    df = st.session_state.df

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat with Data", "📊 Quick Plots", "🔍 Auto Report", "🧹 Clean & Export"])

    # ========================== CHAT TAB ==========================
    with tab1:
        st.subheader("Ask anything in English or Swahili!")

        # Initialize AI
        if OLLAMA_AVAILABLE:
            llm = Ollama(model=model_name)
            sdf = SmartDataframe(df, config={"llm": llm, "enable_cache": False})
        
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("e.g. 'average salary by department', 'top 10 customers', 'plot sales trend'"):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if OLLAMA_AVAILABLE:
                        try:
                            result = sdf.chat(prompt)
                            if isinstance(result, str) and "plot" in result.lower():
                                st.image("exports/charts/last_chart.png", use_container_width=True)
                            elif hasattr(result, 'head'):  # DataFrame
                                st.dataframe(result)
                            else:
                                st.markdown(result)
                            st.session_state.chat.append({"role": "assistant", "content": str(result)})
                        except Exception as e:
                            st.error("AI had a hiccup. Falling back to rule-based engine.")
                            # fall through to rule-based
                    # ==================== FREE RULE-BASED AI (always works) ====================
                    response = ""
                    pl = prompt.lower()

                    # Basic aggregations
                    if any(word in pl for word in ["average", "mean", "avg"]) and "by" in pl:
                        cols = [c for c in df.columns if c.lower() in pl]
                        if len(cols) >= 2:
                            num_col = next(c for c in df.select_dtypes("number").columns if c.lower() in pl)
                            cat_col = next(c for c in cols if c != num_col)
                            result = df.groupby(cat_col)[num_col].mean().round(2)
                            st.bar_chart(result)
                            response = f"**Average {num_col} by {cat_col}**"
                            st.dataframe(result)

                    elif "top" in pl and any(x in pl for x in ["highest", "most", "biggest"]):
                        cols = df.select_dtypes("number").columns
                        if cols.any():
                            top_col = cols[0]
                            top_n = 10
                            result = df.nlargest(top_n, top_col)[[top_col] + df.columns.tolist()[:3]]
                            st.dataframe(result)
                            response = f"Top {top_n} by {top_col}"

                    elif "plot" in pl or "chart" in pl or "graph" in pl:
                        num_cols = df.select_dtypes("number").columns[:2]
                        if len(num_cols) >= 2:
                            fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title=f"{num_cols[1]} vs {num_cols[0]}")
                            st.plotly_chart(fig)
                            response = "Here’s a quick scatter plot"

                    if response:
                        st.markdown(response)
                        st.session_state.chat.append({"role": "assistant", "content": response})

    # ========================== OTHER TABS (same as before) ==========================
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X", df.columns)
        with col2:
            y = st.selectbox("Y", df.columns, index=min(1, len(df.columns)-1))
        chart = st.selectbox("Type", ["Scatter", "Line", "Bar", "Histogram"])
        if chart == "Scatter":
            st.plotly_chart(px.scatter(df, x, y, color=df.columns[0] if len(df.columns)>2 else None))
        elif chart == "Line":
            st.plotly_chart(px.line(df.sort_values(x), x, y))
        elif chart == "Bar":
            st.plotly_chart(px.bar(df.groupby(x)[y].mean().reset_index(), x, y))
        elif chart == "Histogram":
            st.plotly_chart(px.histogram(df, y))

    with tab3:
        if st.button("Generate Full Auto Report"):
            with st.spinner("Analyzing..."):
                report = ProfileReport(df, explorative=True, title="Your Data Report")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                    report.to_file(f.name)
                    with open(f.name, "r", encoding="utf-8") as html:
                        st.components.v1.html(html.read(), height=1000, scrolling=True)
                    os.unlink(f.name)

    with tab4:
        st.download_button("💾 Download Cleaned CSV", 
                           data=df.to_csv(index=False).encode(),
                           file_name="cleaned_data.csv",
                           mime="text/csv")

else:
    st.info("👆 Upload your file to unlock the magic – completely free!")

st.caption("Built with ❤️ in Nairobi • Runs 100% offline with Ollama + Phi-3")
