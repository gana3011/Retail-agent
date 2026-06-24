@echo off
cd /d "%~dp0"
echo Starting Retail Knowledge Chatbot (LangGraph edition)...
echo.
echo 1. If you have a Groq API key, set it:
echo    set GROQ_API_KEY=gsk_your_key_here
echo.
echo 2. Or enter it in the sidebar when the app loads.
echo.
echo Get a free key at: https://console.groq.com/keys
echo.
pause
streamlit run streamlit_app.py
