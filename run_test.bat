@echo off
setlocal EnableExtensions

call "%~dp0setup.bat"
if errorlevel 1 exit /b 1

cmake --build "%~dp0build-ninja" --target test -j
if errorlevel 1 exit /b 1

"%~dp0build-ninja\Test\test.exe"
exit /b %errorlevel%