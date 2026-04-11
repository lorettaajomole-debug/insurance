@echo off
cd "c:\Users\trose\Insurance Regression"
echo Checking git status...
git status
echo.
echo Git RemoteDetails:
git remote -v
echo.
echo Git Log:
git log --oneline -3
echo.
echo Attempting to push to GitHub...
git push -u origin master
if %errorlevel% equ 0 (
    echo Push successful!
) else (
    echo Push failed with error code: %errorlevel%
)
pause
