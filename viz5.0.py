import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import tempfile
import os
import re

st.set_page_config(page_title="AI Insights • Free & Smart", layout="wide", page_icon="🧠📊")

st.title("🧠 AI Insights – Talk to Your Data (100% Free, No Keys)")
st.markdown("Upload CSV/Excel → ask questions in plain English → get answers, tables & charts. Works offline or on Streamlit Cloud.")

# Sidebar
with st.sidebar:
    st.header("📂 Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    st.divider()
    st.subheader("🛠 Quick Clean")
    drop_na = st.checkbox("Drop missing values rows")
    drop_dup = st.checkbox("Remove duplicates")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.chat_history = []

# Load & clean data
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        df = df_raw.copy()
        if drop_dup:
            df = df.drop_duplicates()
        if drop_na:
            df = df.dropna()

        st.session_state.df = df
        st.sidebar.success(f"Loaded & cleaned: {df.shape[0]:,} rows × {df.shape[1]} cols")
    except Exception as e:
        st.sidebar.error(f"File read error: {str(e)}")

# ────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    tab_chat, tab_viz, tab_report, tab_export = st.tabs(
        ["💬 Ask Questions", "📈 Quick Visuals", "🔍 Auto EDA Report", "🗄 Export"]
    )

    # ───── Chat Tab ─────
    with tab_chat:
        st.subheader("Ask me anything about the data")
        st.caption("Examples: 'average price by category', 'top 10 highest sales', 'plot revenue over time', 'show rows where age > 30', 'correlation between price and rating'")

        # Show history
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        prompt = st.chat_input("Your question...")

        if prompt:
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    lower_prompt = prompt.lower().strip()
                    response = ""
                    extra_content = None

                    # ───── Rule-based patterns (expand as needed) ─────
                    # 1. Average / Mean / Avg by group
                    if re.search(r"(average|mean|avg)\b.*\bby\b", lower_prompt):
                        match = re.search(r"(?:average|mean|avg)\s*(?:of)?\s*(\w+)\s*by\s*(\w+)", lower_prompt)
                        if match:
                            val_col, group_col = match.groups()
                            if val_col in df.columns and group_col in df.columns:
                                try:
                                    result = df.groupby(group_col)[val_col].mean().round(2).sort_values(ascending=False)
                                    response = f"**Average {val_col} by {group_col}:**"
                                    extra_content = result.reset_index()
                                    st.bar_chart(result)
                                except:
                                    response = "Couldn't compute — check if columns are numeric/categorical."
                            else:
                                response = f"Columns '{val_col}' or '{group_col}' not found."

                    # 2. Sum / Total by group
                    elif re.search(r"(sum|total)\b.*\bby\b", lower_prompt):
                        match = re.search(r"(?:sum|total)\s*(?:of)?\s*(\w+)\s*by\s*(\w+)", lower_prompt)
                        if match:
                            val_col, group_col = match.groups()
                            if val_col in df.columns and group_col in df.columns:
                                result = df.groupby(group_col)[val_col].sum().round(2).sort_values(ascending=False)
                                response = f"**Total {val_col} by {group_col}:**"
                                extra_content = result.reset_index()
                                st.bar_chart(result)
                            else:
                                response = "Columns not found."

                    # 3. Top N highest/lowest
                    elif "top" in lower_prompt or "highest" in lower_prompt:
                        n = 10
                        match_n = re.search(r"top\s*(\d+)", lower_prompt)
                        if match_n:
                            n = int(match_n.group(1))
                        num_cols = df.select_dtypes(include="number").columns
                        if num_cols.any():
                            sort_col = num_cols[0]  # default to first numeric
                            df_top = df.nlargest(n, sort_col)
                            response = f"**Top {n} rows by {sort_col}:**"
                            extra_content = df_top
                        else:
                            response = "No numeric columns to sort by."

                    # 4. Plot / Chart / Graph
                    elif any(w in lower_prompt for w in ["plot", "chart", "graph", "visualize"]):
                        num_cols = df.select_dtypes(include="number").columns
                        if len(num_cols) >= 2:
                            fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                                             title=f"{num_cols[1]} vs {num_cols[0]}")
                            st.plotly_chart(fig, use_container_width=True)
                            response = f"Quick scatter plot using first two numeric columns."
                        else:
                            response = "Need at least two numeric columns for a plot."

                    # 5. Correlation
                    elif "correlation" in lower_prompt or "correlate" in lower_prompt:
                        corr = df.corr(numeric_only=True)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)
                        response = "Correlation heatmap (numeric columns only)."

                    # 6. Show rows / filter
                    elif "show" in lower_prompt or "filter" in lower_prompt or "where" in lower_prompt:
                        response = "Filter support coming soon — try more specific aggregation questions for now!"

                    # Fallback
                    else:
                        response = (
                            "Sorry, I didn't understand that question yet.\n\n"
                            "Try these examples:\n"
                            "- average salary by department\n"
                            "- total revenue by region\n"
                            "- top 5 highest profit\n"
                            "- plot sales over time\n"
                            "- correlation between variables"
                        )

                    st.markdown(response)
                    if extra_content is not None:
                        if isinstance(extra_content, pd.Series):
                            st.dataframe(extra_content.reset_index())
                        else:
                            st.dataframe(extra_content)

                    st.session_state.chat_history.append(("assistant", response))

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # ───── Visuals Tab ─────
    with tab_viz:
        st.subheader("Quick Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", df.columns, key="x_viz")
        with col2:
            y_col = st.selectbox("Y-axis", df.columns, index=1 if len(df.columns)>1 else 0, key="y_viz")

        viz_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Box", "Histogram"])
        if x_col and y_col:
            if viz_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col)
            elif viz_type == "Line":
                fig = px.line(df.sort_values(x_col), x=x_col, y=y_col)
            elif viz_type == "Bar":
                fig = px.bar(df.groupby(x_col)[y_col].mean().reset_index(), x=x_col, y=y_col)
            elif viz_type == "Box":
                fig = px.box(df, x=x_col, y=y_col)
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=y_col)
            st.plotly_chart(fig, use_container_width=True)

    # ───── Report Tab ─────
    with tab_report:
        st.subheader("Automatic Data Profile Report")
        if st.button("Generate Full Report (takes 10–60s)"):
            with st.spinner("Generating..."):
                profile = ProfileReport(df, title="Data Profile", explorative=True, minimal=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    profile.to_file(tmp.name)
                    with open(tmp.name, "r", encoding="utf-8") as f:
                        html = f.read()
                st.components.v1.html(html, height=1000, scrolling=True)
                os.unlink(tmp.name)

    # ───── Export Tab ─────
    with tab_export:
        st.subheader("Download Cleaned Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

else:
    st.info("Upload your CSV or Excel file to start asking questions!", icon="⬆️")

st.caption("Built for free use • No API keys • Deployable on Streamlit Cloud")
