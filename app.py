import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai


# Load environment variables
load_dotenv()

# Force disable OpenAI usage if not intended (CrewAI sometimes defaults to it)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "OpenAi-API-KEY"

# Streamlit Page Config
st.set_page_config(
    page_title="Agentic Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("🤖 Agentic Analyst")
    st.markdown("---")
    
    api_key = st.text_input("Enter Google API Key", type="password", value="Gemini-Api-Key")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    
    st.markdown("---")
    st.markdown("### 🛠️ Agents")
    st.markdown("- **Data Analyst**: Finds patterns & trends")
    st.markdown("- **Business Strategist**: Provides actionable advice")

def get_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        verbose=True,
        temperature=0.5,
        google_api_key=api_key
    )

def create_crew(df_head, df_stats, api_key):
    llm = get_llm(api_key)
    
    # 1. Data Analyst Agent
    analyst = Agent(
        role='Senior Data Analyst',
        goal='Analyze the dataset to find trends, outliers, and key patterns.',
        backstory="""You are a veteran data analyst with 10 years of experience 
        in spotting hidden trends in complex datasets. You follow a step-by-step approach to ensure accuracy and completeness.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 2. Business Strategist Agent
    strategist = Agent(
        role='Business Strategist',
        goal='Provide actionable strategic advice based on data analysis.',
        backstory="""You are a top-tier business consultant. You take raw data insights 
        and translate them into clear, actionable business strategies . And try to hit the output accuracy to 90 percent by creating charts and graphs.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Task 1: Analyze Data
    analysis_task = Task(
        description=f"""
        Analyze the following dataset summary and statistics:
        
        Head of Data:
        {df_head}
        
        Statistics:
        {df_stats}
        
        Find the factors that are affecting the data and try to hit the output accuracy to 90 percent by creating charts and graphs.
        Use a Step by step approach to ensure accuracy and completeness.
        the procedure is as follows:
        1. Decompose the data into smaller, more manageable chunks.
        2. Analyze each chunk separately.
        3. Combine the results to get a complete picture.
        """,
        agent=analyst,
        expected_output="A list of 3-5 key trends and patterns found in the data data. And try to increase the profit as much as one can"
    )

    # Task 2: Strategy Recommendation
    strategy_task = Task(
        description=f"""
        Based on the analysis provided by the Data Analyst, create a strategic report.
        Take Steps and measure to increase the efficiency of the Whole ongoing process.
        Try to Search the data and try to hit the output accuracy to near 100 percent by creating charts and graphs.
        Use a Step by step approach to ensure accuracy and completeness.
        the procedure is as follows:
        1. Decompose the data into smaller, more manageable chunks.
        2. Analyze each chunk separately.
        3. Combine the results to get a complete picture.
        """,
        agent=strategist,
        expected_output="A strategic report with 3 concrete actions and their rationale. And the best way to make profit"
    )

    crew = Crew(
        agents=[analyst, strategist],
        tasks=[analysis_task, strategy_task],
        verbose=True,
        process=Process.sequential,
        memory=False, # Disable memory to avoid OpenAI embedding requirement
        cache=True
    )
    
    return crew

# Main UI
st.title("📊 Intelligent Data Analysis Platform")
st.markdown("Upload your data and let our **AI Crew** analyze it for you.")

if uploaded_file is not None:
    # Load Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("File Uploaded Successfully!")
        
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 AI Insights", "📈 Visualizations"])
        
        # TAB 1: Data Overview
        with tab1:
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Statistics")
            st.dataframe(df.describe())
            
        # TAB 2: AI Insights
        with tab2:
            st.markdown("### 🤖 Intelligent Analysis")
            st.markdown("Click the button below to launch the AI Crew.")
            
            if st.button("🚀 Run AI Analysis Crew"):
                if not api_key:
                    st.error("Please enter your Google API Key in the sidebar.")
                else:
                    with st.spinner('🤖 Agents are working on your data...'):
                        # Set the API key in environment for langchain/crewai
                        os.environ["GOOGLE_API_KEY"] = api_key
                        
                        # Prepare data summary for the agents
                        # We can't pass the whole DF, so we pass head and describe
                        df_head_str = df.head(10).to_markdown(index=False)
                        df_stats_str = df.describe().to_markdown()
                        
                        crew = create_crew(df_head_str, df_stats_str, api_key)
                        try:
                            result = crew.kickoff()
                            st.success("Analysis Complete!")
                            st.markdown("### 📝 Strategic Report")
                            with st.container(border=True):
                                st.markdown(result)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            if "404" in str(e) or "not found" in str(e).lower():
                                st.warning("The model 'gemini-flash-latest' was not found. Checking available models for your key...")
                                try:
                                    genai.configure(api_key=api_key)
                                    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                                    st.write("Available Models:", models)
                                except Exception as inner_e:
                                    st.error(f"Could not list models: {inner_e}")
        
        # TAB 3: Visualizations
        with tab3:
            st.subheader("📈 Interactive Dashboard")
            
            # Numeric Columns for plotting
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            # Categorical Columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            if not numeric_cols:
                st.warning("No numeric columns found for plotting.")
            else:
                chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_axis = st.selectbox("X-Axis", all_cols)
                with col2:
                    y_axis = st.selectbox("Y-Axis", numeric_cols)
                with col3:
                    color_col = st.selectbox("Color Grouping (Optional)", ["None"] + cat_cols)
                
                if color_col == "None":
                    color_col = None
                
                if st.button("Generate Chart"):
                    if chart_type == "Bar Chart":
                        fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
                    elif chart_type == "Line Chart":
                        fig = px.line(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} trend over {x_axis}")
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis, color=color_col, title=f"Distribution of {x_axis}")
                    elif chart_type == "Box Plot":
                        fig = px.box(df, x=x_axis, y=y_axis, color=color_col, title=f"Box Plot of {y_axis} by {x_axis}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            st.markdown("---")
            st.subheader("🔥 Correlation Heatmap")
            if st.checkbox("Show Correlation Heatmap"):
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix", height=800)
                st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")
