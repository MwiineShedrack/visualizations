import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re

# ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Insights • Free & Smart",
    layout="wide",
    page_icon="🧠📊",
    initial_sidebar_state="expanded"
)

st.title("🧠 AI Insights – Talk to Your Data")
st.markdown("Upload your CSV or Excel file → ask questions in plain English → get answers, tables & charts. 100% free, no API keys needed.")

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    st.divider()
    st.subheader("🛠 Quick Cleaning")
    drop_na = st.checkbox("Drop rows with missing values", value=False)
    drop_duplicates = st.checkbox("Remove duplicate rows", value=False)

# ────────────────────────────────────────────────
# Session State
# ────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.chat_history = []

# ────────────────────────────────────────────────
# Load and clean data
# ────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        df = df_raw.copy()

        original_shape = df.shape
        if drop_duplicates:
            df = df.drop_duplicates()
        if drop_na:
            df = df.dropna()

        st.session_state.df = df

        change_text = ""
        if df.shape != original_shape:
            change_text = f" (cleaned from {original_shape[0]:,} × {original_shape[1]} to {df.shape[0]:,} × {df.shape[1]})"

        st.sidebar.success(f"Data loaded{change_text}")

    except Exception as e:
        st.sidebar.error(f"Error reading file: {str(e)}")

# ────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    tab_chat, tab_viz, tab_summary, tab_export = st.tabs(
        ["💬 Ask Questions", "📈 Quick Charts", "📊 Data Summary", "⬇️ Export"]
    )

    # ───── Chat / AI Questions Tab ─────
    with tab_chat:
        st.subheader("Ask anything about your data")
        st.caption(
            "Try examples:\n"
            "• average price by category\n"
            "• total sales by region\n"
            "• top 10 highest revenue\n"
            "• plot profit vs cost\n"
            "• correlation between variables\n"
            "• show top categories"
        )

        # Display chat history
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        # Input
        user_input = st.chat_input("Your question...")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    lower_input = user_input.lower().strip()
                    response = ""
                    display_df = None
                    display_chart = None

                    # ───── Pattern matching rules ─────

                    # Average / mean / avg by group
                    if re.search(r"(average|mean|avg)\b.*\bby\b", lower_input):
                        match = re.search(r"(?:average|mean|avg)\s*(?:of)?\s*([\w\s]+?)\s*by\s*([\w\s]+)", lower_input)
                        if match:
                            val_part, group_part = match.groups()
                            val_col = val_part.strip()
                            group_col = group_part.strip()
                            if val_col in df.columns and group_col in df.columns:
                                try:
                                    result = df.groupby(group_col)[val_col].mean().round(2).sort_values(ascending=False)
                                    response = f"**Average {val_col} by {group_col}** (sorted descending)"
                                    display_df = result.reset_index()
                                    display_chart = px.bar(display_df, x=group_col, y=val_col, title=response)
                                except:
                                    response = "Could not calculate — check if the value column is numeric."
                            else:
                                response = f"Columns '{val_col}' or '{group_col}' not found in dataset."

                    # Sum / total by group
                    elif re.search(r"(sum|total)\b.*\bby\b", lower_input):
                        match = re.search(r"(?:sum|total)\s*(?:of)?\s*([\w\s]+?)\s*by\s*([\w\s]+)", lower_input)
                        if match:
                            val_part, group_part = match.groups()
                            val_col = val_part.strip()
                            group_col = group_part.strip()
                            if val_col in df.columns and group_col in df.columns:
                                result = df.groupby(group_col)[val_col].sum().round(2).sort_values(ascending=False)
                                response = f"**Total {val_col} by {group_col}** (sorted descending)"
                                display_df = result.reset_index()
                                display_chart = px.bar(display_df, x=group_col, y=val_col, title=response)
                            else:
                                response = f"Columns '{val_col}' or '{group_col}' not found."

                    # Top N
                    elif any(w in lower_input for w in ["top", "highest", "largest", "most"]):
                        n = 10
                        m = re.search(r"(?:top|highest|largest)\s*(\d+)", lower_input)
                        if m:
                            n = int(m.group(1))
                        num_cols = df.select_dtypes(include="number").columns
                        if len(num_cols) > 0:
                            sort_col = num_cols[0]
                            top_df = df.nlargest(n, sort_col)
                            response = f"**Top {n} rows by {sort_col}**"
                            display_df = top_df
                        else:
                            response = "No numeric columns available to rank."

                    # Plot / chart / visualize
                    elif any(w in lower_input for w in ["plot", "chart", "graph", "visualize", "show trend"]):
                        num_cols = df.select_dtypes(include="number").columns
                        if len(num_cols) >= 2:
                            x = num_cols[0]
                            y = num_cols[1] if len(num_cols) > 1 else num_cols[0]
                            display_chart = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
                            response = f"Showing scatter plot of first two numeric columns: **{y} vs {x}**"
                        else:
                            response = "Need at least two numeric columns to create a plot."

                    # Correlation
                    elif "correlation" in lower_input or "correlate" in lower_input:
                        numeric = df.select_dtypes(include="number")
                        if numeric.shape[1] >= 2:
                            corr = numeric.corr()
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                            st.pyplot(fig)
                            response = "**Correlation heatmap** (numeric columns only)"
                        else:
                            response = "Not enough numeric columns for correlation analysis."

                    # Fallback
                    if not response:
                        response = (
                            "Sorry, I didn't quite understand that question.\n\n"
                            "**Try asking something like:**\n"
                            "- average salary by department\n"
                            "- total revenue by region\n"
                            "- top 10 highest sales\n"
                            "- plot profit vs expenses\n"
                            "- correlation between price and rating"
                        )

                    st.markdown(response)

                    if display_df is not None:
                        st.dataframe(display_df, use_container_width=True)

                    if display_chart is not None:
                        st.plotly_chart(display_chart, use_container_width=True)

                    st.session_state.chat_history.append(("assistant", response))

        # Clear button
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    # ───── Quick Charts Tab ─────
    with tab_viz:
        st.subheader("Custom Visualizations")

        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X axis", df.columns, key="x_axis")
        with col2:
            y_axis = st.selectbox("Y axis", df.columns, index=min(1, len(df.columns)-1), key="y_axis")

        chart_type = st.selectbox("Chart type", ["Scatter", "Line", "Bar", "Box", "Histogram"])

        if x_axis and y_axis:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            elif chart_type == "Line":
                fig = px.line(df.sort_values(x_axis), x=x_axis, y=y_axis)
            elif chart_type == "Bar":
                agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
                fig = px.bar(agg_df, x=x_axis, y=y_axis)
            elif chart_type == "Box":
                fig = px.box(df, x=x_axis, y=y_axis)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=y_axis)
            st.plotly_chart(fig, use_container_width=True)

    # ───── Data Summary Tab ─────
    with tab_summary:
        st.subheader("Quick Data Summary")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Basic Information**")
            st.write(f"Rows: {df.shape[0]:,}")
            st.write(f"Columns: {df.shape[1]}")
            st.write("Data types:")
            st.write(df.dtypes.value_counts().to_frame(name="Count"))

        with colB:
            st.markdown("**Missing Values**")
            miss = df.isnull().sum()
            miss_pct = (miss / len(df) * 100).round(2)
            miss_table = pd.DataFrame({
                "Missing": miss,
                "% Missing": miss_pct
            }).sort_values("Missing", ascending=False)
            st.dataframe(miss_table.style.format({"% Missing": "{:.2f}%"}))

        st.subheader("Numeric Columns Overview")
        st.dataframe(df.describe().T.style.format("{:.2f}"))

        st.subheader("Explore a Column")
        selected_col = st.selectbox("Select column", df.columns, key="explore_col")
        if selected_col:
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                fig = px.histogram(df, x=selected_col, marginal="box", title=f"Distribution: {selected_col}")
            else:
                counts = df[selected_col].value_counts().head(12)
                fig = px.bar(counts, x=counts.index, y=counts.values, title=f"Top values: {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

    # ───── Export Tab ─────
    with tab_export:
        st.subheader("Download Cleaned Dataset")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            help="Downloads the currently loaded (and cleaned) data"
        )

else:
    st.info("Please upload a CSV or Excel file to begin.", icon="⬆️")

st.caption("Built with ❤️ in Nairobi • 100% free • No external APIs or keys required")
