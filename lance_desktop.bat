@echo off
title Vision IA - Mode Desktop Natif
cd /d "%~dp0"
echo ============================================================
echo   [VISION IA TEMPS REEL] - LANCEMENT DESKTOP (OPENCV)
echo ============================================================
echo.
.\venv\Scripts\python.exe desktop.py
if errorlevel 1 (
    echo.
    echo Une erreur est survenue lors de l'execution.
    pause
)
