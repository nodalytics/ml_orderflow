@echo off
set "SOURCE_DIR=."
set "OUTPUT_ZIP=ml_orderflow.zip"

echo Creating package.zip...

REM Create a temporary directory to hold the files to be zipped
mkdir temp_package

REM Copy nautilus folder excluding .venv
echo .venv\ > exclude.txt
echo dist\ >> exclude.txt
echo build\ >> exclude.txt
echo __pycache__\ >> exclude.txt
xcopy "%SOURCE_DIR%\nautilus" "temp_package\nautilus\" /E /I /Q /EXCLUDE:exclude.txt

REM Copy sandbox folder excluding .venv
xcopy "%SOURCE_DIR%\sandbox" "temp_package\sandbox\" /E /I /Q /EXCLUDE:exclude.txt

REM Zip the contents of the temporary directory
powershell -command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory('temp_package', '%OUTPUT_ZIP%');"

REM Clean up temporary directory and exclude files
rmdir /S /Q temp_package
del exclude.txt

echo Package created successfully: %OUTPUT_ZIP%
pause
