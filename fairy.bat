@echo off
taskkill /f /im chrome.exe 2>nul
cd /d "%~dp0"
set FAIRY_USE_HERMES=1
python fairy.py
