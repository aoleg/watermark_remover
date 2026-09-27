@echo off
setlocal EnableDelayedExpansion
rem ---------------------------------------------------------------------------
rem recursive_scan.bat <folder> [watermark_remover.py parameters]
rem
rem Runs watermark_remover.py once for every first-level subfolder of <folder>.
rem For a subfolder  <folder>\NAME  the output goes to  <folder>\_NAME.
rem Every parameter after <folder> is passed through unchanged, e.g.
rem     recursive_scan.bat D:\datasets --png -R -w yolov12x-dino3-watermark-detection.pt
rem
rem Subfolders whose name starts with "_" are skipped, so re-running the command
rem does not process previous output folders. Each output folder keeps its own
rem .processing_log.txt, so an interrupted run resumes where it stopped.
rem
rem The script runs from this .bat file's directory so the default weights file
rem resolves; give relative paths in the parameters relative to that directory.
rem ---------------------------------------------------------------------------

set "HERE=%~dp0"
if "%~1"=="" (
    echo Usage: %~nx0 ^<folder^> [watermark_remover.py parameters]
    exit /b 1
)
if not exist "%~1\" (
    echo ERROR: folder not found: %~1
    exit /b 1
)
set "ROOT=%~f1"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem Collect every parameter after the folder. (shift also moves %0, hence HERE above.)
set "ARGS="
:collect
shift
if "%~1"=="" goto collected
set "ARGS=!ARGS! %1"
goto collect
:collected

pushd "%HERE%"
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

set /a COUNT=0
for /d %%D in ("%ROOT%\*") do (
    set "NAME=%%~nxD"
    if not "!NAME:~0,1!"=="_" (
        set /a COUNT+=1
        echo.
        echo ===== [!COUNT!] %%~fD  -^>  %ROOT%\_!NAME!
        "%PY%" watermark_remover.py -i "%%~fD" -o "%ROOT%\_!NAME!"!ARGS!
        if errorlevel 1 (
            echo ===== watermark_remover.py exited with code !errorlevel! for %%~fD
        )
    )
)
popd

if %COUNT%==0 echo No subfolders found in %ROOT%
echo.
echo ===== Done: %COUNT% subfolder(s) processed.
endlocal
