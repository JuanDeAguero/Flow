@echo off
setlocal EnableExtensions

call "%~dp0setup.bat"
if errorlevel 1 exit /b 1

cmake --build "%~dp0build-ninja" --target mnist -j
if errorlevel 1 exit /b 1

pushd "%~dp0Showcase\MNIST"
"%~dp0build-ninja\Showcase\MNIST\mnist.exe"
set "MNIST_EXIT=%errorlevel%"
popd

exit /b %MNIST_EXIT%