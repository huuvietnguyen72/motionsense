@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
py -3.13 -X utf8 --version >nul 2>&1
if errorlevel 1 (
  echo Chưa có CPython Windows 3.13. Cài bản 64-bit từ python.org, gồm Python Launcher.
  echo Sau đó mở lại cửa sổ này và chạy Cai_dat.bat.
  pause
  exit /b 2
)
py -3.13 -X utf8 "%~dp0scripts\install.py"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
