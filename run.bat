@echo off
setlocal enabledelayedexpansion

:: Set the Python command (modify if needed)
set PYTHON_CMD=python

:: Check if Python is installed
where %PYTHON_CMD% >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the system PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Ask user if they want to recreate the virtual environment
set /p RECREATE_VENV="Do you want to recreate the virtual environment? (y/n): "

if /i "%RECREATE_VENV%"=="y" (
    if exist "venv" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
        if !errorlevel! neq 0 (
            echo Failed to remove existing virtual environment.
            pause
            exit /b 1
        )
    )

    echo Creating a new virtual environment...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment has been recreated successfully.
) else (
    if not exist "venv" (
        echo Virtual environment does not exist. Creating a new one...
        %PYTHON_CMD% -m venv venv
        if %errorlevel% neq 0 (
            echo Failed to create virtual environment.
            pause
            exit /b 1
        )
        echo New virtual environment has been created.
    ) else (
        echo Using existing virtual environment without changes.
    )
)

echo Activating the virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo Installing the necessary requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo Launching OllamaGradio UI...
%PYTHON_CMD% geeky-Web-ui-main.py
if %errorlevel% neq 0 (
    echo Failed to launch OllamaGradio UI.
    pause
    exit /b 1
)

echo.
echo Script completed successfully.
pause
