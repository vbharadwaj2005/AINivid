@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
echo Starting AI Nivid Streamlit...
streamlit run app.py
pause

