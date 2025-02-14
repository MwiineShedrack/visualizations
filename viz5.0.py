import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import tempfile
import re

# App Title
st.set_page_config(page_title="Mwiine's Data Insights", layout="wide")
st.title("📊 Ultimate Data Visualization App")
st.sidebar.header("Upload Your Dataset")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Natural Language Query Section
    st.sidebar.subheader("🗣️ Ask a Question about Your Data")
    query = st.sidebar.text_input("Enter your question (e.g., What is the average sales per region?)")

    if query:
        # Function to process the query and respond with appropriate data
        def process_query(query):
            query = query.lower()  # Convert query to lowercase for simplicity
            if "average" in query and "per" in query:
                # Extract column names (assuming question like "What is the average sales per region?")
                match = re.search(r"average of (\w+) per (\w+)", query)
                if match:
                    column1, column2 = match.groups()
                    if column1 in df.columns and column2 in df.columns:
                        result = df.groupby(column2)[column1].mean()
                        return f"The average {column1} per {column2}:\n{result}"
                    else:
                        return "Sorry, the columns you asked for do not exist in the dataset."
                else:
                    return "Please use the correct format, e.g., 'average of sales per region'."

            elif "sum" in query and "per" in query:
                # Extract column names for sum calculation (e.g., "What is the total sales per region?")
                match = re.search(r"sum of (\w+) per (\w+)", query)
                if match:
                    column1, column2 = match.groups()
                    if column1 in df.columns and column2 in df.columns:
                        result = df.groupby(column2)[column1].sum()
                        return f"The total {column1} per {column2}:\n{result}"
                    else:
                        return "Sorry, the columns you asked for do not exist in the dataset."
                else:
                    return "Please use the correct format, e.g., 'sum of sales per region'."

            # More queries can be added as needed (e.g., median, min, max, etc.)

            return "Sorry, I couldn't understand that query. Try asking something like 'What is the average of sales per region?'"

        # Show response for the query
        response = process_query(query)
        st.sidebar.write(response)

    # AI-Powered Insights (Enhanced)
    st.sidebar.subheader("🔍 AI-Powered Insights")
    
    if st.sidebar.button("Generate AI Insights"):
        missing_values = df.isnull().sum().sum()
        duplicated_rows = df.duplicated().sum()
        summary = f"🔹 **Missing Values:** {missing_values}\n🔹 **Duplicated Rows:** {duplicated_rows}\n"
        summary = summary.encode("ascii", "ignore").decode()  # Removes non-ASCII characters

        # Generating a profiling report
        pr = ProfileReport(df, explorative=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            pr.to_file(tmpfile.name)
            with open(tmpfile.name, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=800, scrolling=True)
        
        st.download_button(label="📥 Download AI Summary", data=summary, file_name="ai_summary.txt")

    # Customizable Dashboards with Multi-Variable Selection
    st.sidebar.subheader("📊 Comparative Visualization")
    chart_type = st.sidebar.selectbox("Choose a Chart Type", ["Scatter", "Bar", "Line", "Heatmap"])
    x_axes = st.sidebar.multiselect("Select X-axis", df.columns)
    y_axes = st.sidebar.multiselect("Select Y-axis", df.columns)

    if x_axes and y_axes:
        for x in x_axes:
            for y in y_axes:
                fig = None
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x, y=y, title=f"Scatter Plot: {x} vs {y}")
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x, y=y, title=f"Bar Chart: {x} vs {y}")
                elif chart_type == "Line":
                    fig = px.line(df, x=x, y=y, title=f"Line Chart: {x} vs {y}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
