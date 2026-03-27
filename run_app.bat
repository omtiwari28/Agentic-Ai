@echo off
title AI-Powered Business Intelligence Platform
echo Starting the application...

:: Change to the directory where the batch script is located
cd /d "%~dp0"

:: Activate the virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run the Streamlit app
cd Data_Analyser_App
echo Launching Streamlit...
python -m streamlit run app.py

:: Keep the window open if there's an error
pause
