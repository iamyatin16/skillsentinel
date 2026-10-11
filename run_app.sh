#!/bin/bash
cd "$(dirname "$0")"
PY=python3; command -v python3 >/dev/null || PY=python
[ -d venv ] || $PY -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python check_setup.py
streamlit run app.py
