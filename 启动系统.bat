@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set "PYTHONPATH=%~dp0.deps;%PYTHONPATH%"
set "STUDIO_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%STUDIO_PYTHON%" goto python_ready
set "STUDIO_PYTHON=python"
where python >nul 2>&1
if errorlevel 1 (
    echo 未找到 Python，请按 README 安装。
    pause
    exit /b 1
)
:python_ready
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo 未找到 FFmpeg，请按 README 安装。
    pause
    exit /b 1
)
where ffprobe >nul 2>&1
if errorlevel 1 (
    echo 未找到 FFprobe，请按 README 安装。
    pause
    exit /b 1
)
"%STUDIO_PYTHON%" studio.py
if errorlevel 1 pause
