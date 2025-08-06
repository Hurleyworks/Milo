@echo off
:: Script to build and run AreaLightHandlerTest

echo Building AreaLightHandlerTest...
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" ^
    unittest\builds\VisualStudio2022\projects\AreaLightHandlerTest.vcxproj ^
    /p:Configuration=Debug /p:Platform=x64

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Running AreaLightHandlerTest...
builds\bin\Debug-windows-x86_64\AreaLightHandlerTest\AreaLightHandlerTest.exe

if %ERRORLEVEL% NEQ 0 (
    echo Tests failed!
    exit /b %ERRORLEVEL%
)

echo All tests passed!