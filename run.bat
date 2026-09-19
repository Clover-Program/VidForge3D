@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

start chrome.exe --kiosk --user-data-dir="%TEMP%\kiosk_profile" "http://localhost:8080"
@powershell -NoProfile -NoExit -ExecutionPolicy Bypass -File "%~dp0server.ps1"
pause