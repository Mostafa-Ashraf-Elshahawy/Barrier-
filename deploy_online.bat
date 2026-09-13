@echo off
echo ======================================================================
echo 🛡️ Launching Barrier Streamlit App with Free Public Cloudflare Domain
echo ======================================================================
echo.

:: Check if cloudflared exists
if not exist cloudflared.exe (
    echo Downloading Cloudflare tunnel binary...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
)

:: Start Streamlit server in background
echo 1. Starting Streamlit application on http://localhost:8501...
start "Barrier Streamlit Server" cmd /c "streamlit run app.py"

:: Wait for server to initialize
timeout /t 5 /nobreak >nul

:: Start Public Tunnel
echo 2. Launching Public HTTPS Tunnel...
echo Look for your public URL below:
echo.
.\cloudflared.exe tunnel --url http://localhost:8501
