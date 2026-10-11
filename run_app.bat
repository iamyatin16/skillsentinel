@echo off
REM Double-click on Windows. Sets up (first time only) and opens SkillSentinel.
if not exist venv ( python -m venv venv )
call venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py
streamlit run app.py
pause
