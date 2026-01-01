@echo off
set "FLOW_ROOT=%~dp0"
if "%FLOW_ROOT:~-1%"=="\" set "FLOW_ROOT=%FLOW_ROOT:~0,-1%"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo ERROR: vswhere not found at "%VSWHERE%".
  echo Install Visual Studio with C++ workloads.
  exit /b 1
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
if not defined VSINSTALL (
  echo ERROR: Could not find a Visual Studio installation with MSVC tools.
  exit /b 1
)

set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
  echo ERROR: VsDevCmd.bat not found at "%VSDEVCMD%".
  exit /b 1
)

call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 (
  echo ERROR: Failed to initialize Visual Studio developer environment.
  exit /b 1
)

if not defined FLOW_CUDA_ROOT set "FLOW_CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
if not exist "%FLOW_CUDA_ROOT%\bin\nvcc.exe" (
  echo ERROR: nvcc.exe not found at "%FLOW_CUDA_ROOT%\bin\nvcc.exe".
  echo Set FLOW_CUDA_ROOT to a CUDA Toolkit folder that contains bin\nvcc.exe.
  exit /b 1
)

set "PATH=%FLOW_CUDA_ROOT%\bin;%PATH%"

set "NINJA_EXE="
for /f "delims=" %%P in ('where ninja 2^>nul') do (
  set "NINJA_EXE=%%P"
  goto :ninja_found
)

for /f "delims=" %%N in ('dir /b /s "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Ninja-build.Ninja_*\ninja.exe" 2^>nul') do (
  set "NINJA_EXE=%%N"
  goto :ninja_found
)

:ninja_found
if not defined NINJA_EXE (
  echo ERROR: Ninja not found.
  echo Install it with: winget install --id Ninja-build.Ninja -e
  exit /b 1
)

set "FLOW_CUDA_ROOT_FWD=%FLOW_CUDA_ROOT:\=/%"
set "FLOW_NVCC_FWD=%FLOW_CUDA_ROOT_FWD%/bin/nvcc.exe"

if not exist "%FLOW_ROOT%\build-ninja" mkdir "%FLOW_ROOT%\build-ninja"

cmake -S "%FLOW_ROOT%" -B "%FLOW_ROOT%\build-ninja" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" -DFLOW_CUDA_ROOT="%FLOW_CUDA_ROOT_FWD%" -DCMAKE_CUDA_COMPILER="%FLOW_NVCC_FWD%" -DFLOW_CUDA_ALLOW_UNSUPPORTED_COMPILER=ON

if errorlevel 1 (
  echo ERROR: CMake configure failed.
  exit /b 1
)

echo OK: Configured build in "%FLOW_ROOT%\build-ninja".
exit /b 0