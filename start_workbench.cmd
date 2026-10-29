@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m pickup_measure.workbench
) else (
    python -m pickup_measure.workbench
)
pause
