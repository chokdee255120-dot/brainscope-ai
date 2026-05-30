@echo off
echo Starting BrainScope AI...
echo.
echo Open your browser and go to:
echo   http://localhost:8000
echo.
cd /d "%~dp0backend"
pip install -r requirements.txt -q
python -m uvicorn main:app --reload --port 8000
pause
