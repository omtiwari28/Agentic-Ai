# 🤖 Agentic Data Analyst — AI-Powered Business Intelligence Platform

> A multi-agent Streamlit application that takes any CSV or Excel file and delivers structured business intelligence — powered by a **CrewAI** pipeline of two specialized AI agents running on **Google Gemini**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit) ![CrewAI](https://img.shields.io/badge/CrewAI-Agentic-purple) ![Gemini](https://img.shields.io/badge/Google-Gemini-orange?logo=google)

---

## 🎯 What It Does

Upload any CSV or Excel dataset and the platform will:

1. **Preview** your data and show descriptive statistics instantly
2. **Launch a 2-agent AI Crew** that analyzes patterns and generates a full strategic report
3. **Visualize** your data through an interactive chart builder with 5 chart types
4. **Answer questions** about your dataset in natural language via a chat interface
5. **Export** both the cleaned data and the AI-generated report as downloadable files

---

## 🤖 The AI Crew

This app uses a **sequential multi-agent pipeline** built with CrewAI:

| Agent | Role | Responsibility |
|-------|------|----------------|
| **Senior Data Analyst** | Pattern Recognition | Identifies trends, outliers, and key variables across the dataset |
| **Business Strategist** | Strategic Advisory | Translates data insights into actionable business recommendations |

The agents run sequentially — the Strategist receives the Analyst's output and builds on it, producing a layered report that combines raw data findings with business context.

---

## 🖥️ App Features

### 📊 Tab 1 — Data Overview
- Live data preview table
- Full descriptive statistics (`df.describe()`)
- One-click CSV export of the loaded dataset

### 🤖 Tab 2 — AI Insights
- Triggers the 2-agent CrewAI pipeline on demand
- Displays a structured markdown report with Data Summary, Key Insights, and Business Recommendations
- Downloadable strategic report as a `.txt` file

### 📈 Tab 3 — Interactive Visualizations
- Chart builder supporting Bar, Line, Scatter, Histogram, and Box Plot
- Dynamic axis and color grouping selectors based on your dataset's columns
- Correlation heatmap for numeric variables

### 💬 Tab 4 — Chat with Data
- Ask any natural language question about your dataset
- Powered by Gemma 3 27B for conversational data Q&A
- Dataset schema and sample rows are injected into the prompt for accurate answers

---

## 🔁 Architecture

```
User uploads CSV/Excel
        │
        ▼
Streamlit UI (4 tabs)
        │
        ├── Tab 1: Pandas preview + stats
        │
        ├── Tab 2: CrewAI Pipeline
        │           │
        │           ├── Agent 1: Senior Data Analyst (Gemini Flash)
        │           │   └── Task: Analyze trends, outliers, patterns
        │           │
        │           └── Agent 2: Business Strategist (Gemini Flash)
        │               └── Task: Strategic recommendations
        │
        ├── Tab 3: Plotly chart builder + correlation heatmap
        │
        └── Tab 4: LLM Q&A (Gemma 3 27B)
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Web application framework |
| **CrewAI** | Multi-agent orchestration |
| **LangChain Google GenAI** | LLM interface for Gemini |
| **Google Gemini Flash** | Powers the analyst and strategist agents |
| **Google Gemma 3 27B** | Powers the chat Q&A tab |
| **Plotly Express** | Interactive visualizations |
| **Pandas** | Data loading and processing |
| **python-dotenv** | Environment variable management |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/agentic-data-analyst.git
cd agentic-data-analyst
```

### Step 2 — Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure your API key
```bash
cp .env.example .env
```
Open `.env` and replace `your_google_api_key_here` with your actual Gemini API key.

### Step 5 — Run the app
```bash
streamlit run app.py
```

Or on Windows, double-click `run_app.bat`.

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
📂 agentic-data-analyst/
├── 📄 app.py                  ← Main Streamlit application
├── 📄 check_models.py         ← Utility to list available Gemini models
├── 📄 requirements.txt        ← Python dependencies
├── 📄 run_app.bat             ← Windows one-click launcher
├── 📄 .env.example            ← Environment variable template
├── 📄 .gitignore              ← Excludes .env and other sensitive files
└── 📄 README.md               ← You are here
```

---

## 🔐 Security

- Your API key is entered in the sidebar at runtime and is **never stored or logged**
- The `.env` file is excluded from version control via `.gitignore`
- Never commit your actual `.env` file or hardcode API keys in source files

---

## 💡 Potential Improvements

- **Add more agents** — a dedicated Visualization Agent that auto-generates the most relevant charts based on data type
- **Support more file types** — JSON, Parquet, Google Sheets URL as input
- **Memory across sessions** — persist conversation history in the Chat tab using `st.session_state`
- **Deploy to Streamlit Cloud** — one-click public sharing with secrets management
- **Agent tool use** — give the Data Analyst agent access to a Python REPL tool so it can run actual computations rather than reasoning from a data sample

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Built with ❤️ using Streamlit, CrewAI, and Google Gemini.  
*Part of my AI automation portfolio — feedback welcome!*
