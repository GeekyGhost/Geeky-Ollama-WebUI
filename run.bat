@echo off
if exist "venv" (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

echo Creating a new virtual environment...
python -m venv venv

echo Activating the virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing the necessary requirements...
pip install -r requirements.txt

echo Launching OllamaGradio UI
python geeky-Web-ui-main.py

pause
