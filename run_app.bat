@echo off
setlocal
cd /d "%~dp0"

set "BASE_PYTHON="
if exist "C:\anacoda3\python.exe" set "BASE_PYTHON=C:\anacoda3\python.exe"
if not defined BASE_PYTHON if exist "C:\anaconda3\python.exe" set "BASE_PYTHON=C:\anaconda3\python.exe"
if not defined BASE_PYTHON where python >nul 2>nul && set "BASE_PYTHON=python"
if not defined BASE_PYTHON if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "BASE_PYTHON=%LocalAppData%\Programs\Python\Python312\python.exe"

if not defined BASE_PYTHON (
    echo Python was not found. Install Python 3.12, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating an isolated project environment. This only happens once...
    "%BASE_PYTHON%" -m venv .venv
    if errorlevel 1 (
        echo Could not create the project environment.
        pause
        exit /b 1
    )
)

set "PROJECT_PYTHON=%CD%\.venv\Scripts\python.exe"
echo Using project Python: %PROJECT_PYTHON%
"%PROJECT_PYTHON%" -c "import streamlit, emoji, joblib, sklearn; assert sklearn.__version__ == '1.6.1'" >nul 2>nul
if errorlevel 1 (
    echo Installing compatible project packages. The first run can take a few minutes...
    set "PIP_CACHE_DIR=%CD%\.pip-cache"
    "%PROJECT_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed. Please read README.md.
        pause
        exit /b 1
    )
)

"%PROJECT_PYTHON%" -m streamlit run app.py --client.toolbarMode minimal
if errorlevel 1 pause

