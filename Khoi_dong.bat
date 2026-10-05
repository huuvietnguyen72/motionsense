@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Chưa có môi trường. Hãy chạy Cai_dat.bat trước.
  pause
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0scripts\launch.py" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
