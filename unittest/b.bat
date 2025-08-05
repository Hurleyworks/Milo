@echo off
:: Build script for UnitTest projects
msbuild builds\VisualStudio2022\UnitTests.sln /p:Configuration=Debug /p:Platform=x64