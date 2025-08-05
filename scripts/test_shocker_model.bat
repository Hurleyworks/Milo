@echo off
REM test_shocker_model.bat - Compile and run ShockerModelTest unit tests

echo ========================================
echo Building ShockerModelTest...
echo ========================================

"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" unittest\builds\VisualStudio2022\projects\ShockerModelTest.vcxproj /p:Configuration=Debug /p:Platform=x64

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Build succeeded! Running tests...
echo ========================================
echo.

builds\bin\Debug-windows-x86_64\ShockerModelTest\ShockerModelTest.exe

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo TESTS FAILED!
    echo ========================================
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo ALL TESTS PASSED!
echo ========================================