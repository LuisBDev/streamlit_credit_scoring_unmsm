@echo off
REM ======================================================
REM Script para iniciar la aplicacion de Credit Scoring
REM Sistema operativo: Windows
REM ======================================================

echo ===============================================
echo  Sistema de Credit Scoring - Microfinanzas
echo ===============================================
echo.

REM Verificar si existe el entorno virtual
if not exist "venv" (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
    echo [OK] Entorno virtual creado
    echo.
)

REM Activar entorno virtual
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat

REM Actualizar pip primero
echo [INFO] Actualizando pip...
python -m pip install --upgrade pip

REM Verificar si las dependencias estan instaladas
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [INFO] Instalando dependencias...
    pip install --upgrade -r requirements.txt
    echo [OK] Dependencias instaladas
    echo.
) else (
    echo [INFO] Verificando versiones de dependencias...
    pip install --upgrade -r requirements.txt
    echo [OK] Dependencias actualizadas
    echo.
)

REM Verificar que existe el modelo
if not exist "results\models\credit_scoring_model_20250917_154932.pkl" (
    echo [ERROR] No se encontro el modelo entrenado
    echo [ERROR] Ruta esperada: results\models\credit_scoring_model_20250917_154932.pkl
    echo.
    pause
    exit /b 1
)

REM Iniciar aplicacion
echo [INFO] Iniciando aplicacion Streamlit...
echo [INFO] La aplicacion se abrira en http://localhost:8501
echo [INFO] Presiona Ctrl+C para detener
echo.
streamlit run app_credit_scoring.py

pause

