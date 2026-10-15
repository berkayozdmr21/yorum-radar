@echo off
cd /d "%~dp0"
title Yorum Radar
echo Yorum Radar hazirlaniyor...
where python >nul 2>&1
if errorlevel 1 (
  echo Python bulunamadi. Python 3.10 veya ustunu kurun.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Baslatma tamamlanamadi. Yukaridaki hatayi veya baslatma.log dosyasini kontrol edin.
pause
