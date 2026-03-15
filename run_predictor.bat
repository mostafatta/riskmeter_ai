@echo off
color 0A
title Tadawul Risk Analyzer - Full Setup & Run

echo ========================================================
echo      SAUDI MARKET RISK ANALYZER - FULL PIPELINE
echo ========================================================
echo.
echo [STEP 1] Checking and Installing Libraries...
pip install -r requirements.txt
echo.
echo --------------------------------------------------------
echo.
echo [STEP 2] Downloading Market Data ^& Generating 500 Portfolios...
echo (This might take a minute to fetch 20 stocks from Yahoo Finance)
python src/data_generator.py
echo.
echo --------------------------------------------------------
echo.
echo [STEP 3] Training the Artificial Intelligence Model...
python src/ml_model.py
echo.
echo --------------------------------------------------------
echo.
echo [STEP 4] Launching the Interactive Predictor...
echo.
python predict_risk.py

echo.
echo ========================================================
echo All tasks completed. You can close this window.
pause