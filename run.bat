@echo off
if not exist "venv" (
    echo Creating a new virtual environment...
    python -m venv venv
)

echo Activating the virtual environment...
call venv\Scripts\activate

echo Installing the necessary requirements...
pip install -r requirements.txt

echo Launching OllamaGradio UI
geeky-Web-ui-main.py

pause
