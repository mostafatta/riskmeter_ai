import sys
import os

# --- Fix Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_DIR)
sys.path.append(BASE_DIR)

import joblib
import pandas as pd
import numpy as np

# === تم حل مشكلة الخطوط الصفراء بإضافة .src ===
from src.data_loader import TadawulDataLoader
from src.calculations import RiskCalculator
from src.risk_labeler import RiskLabeler


def get_user_portfolio():
    """Get portfolio tickers and weights from user input."""
    tickers, weights = [], []
    print("\n" + "=" * 45)
    print("       NEW PORTFOLIO ENTRY")
    print("=" * 45)

    try:
        num_stocks = int(input(
            "How many stocks in your portfolio? (e.g. 3): "
        ))
        if num_stocks <= 0:
            print("Error: Must have at least 1 stock.")
            return None, None
    except ValueError:
        print("Error: Please enter a valid number.")
        return None, None

    remaining_weight = 100.0

    for i in range(num_stocks):
        ticker = input(
            f"\n  Stock #{i+1} Ticker (e.g., 2222.SR): "
        ).strip()
        if len(ticker) > 0 and not ticker.upper().endswith('.SR'):
            ticker += ".SR"

        raw_weight = input(
            f"  Weight % (Remaining: {remaining_weight}%): "
        ).replace('%', '').strip()
        w_input = float(raw_weight) if raw_weight else 0.0

        tickers.append(ticker.upper())
        weights.append(w_input / 100.0)
        remaining_weight -= w_input

    # Show summary
    print(f"\n  Portfolio Summary:")
    print(f"  {'Ticker':<15}{'Weight':<10}")
    print(f"  {'-'*20}")
    for t, w in zip(tickers, weights):
        print(f"  {t:<15}{w*100:.1f}%")

    return tickers, weights


def process_prediction(tickers, weights):
    """Process calculation and AI prediction."""
    print(f"\n{'-'*45}")
    print("  PROCESSING... PLEASE WAIT")
    print(f"{'-'*45}")

    try:
        data_directory = os.path.join(BASE_DIR, 'data', 'raw')
        
        # Load from Yahoo
        loader = TadawulDataLoader(tickers=tickers, data_dir=data_directory)
        loader.fetch_stock_data()
        loader.fetch_market_data()

        meta_path = os.path.join(loader.data_dir, "stocks_metadata.csv")
        if not os.path.exists(meta_path):
            loader.fetch_metadata()
        meta_df = pd.read_csv(meta_path).set_index("Ticker")

        # Calculations
        calc = RiskCalculator(data_dir=data_directory)
        calc.load_data()
        calc.calculate_daily_returns()

        metrics = calc.calculate_portfolio_risk(weights)
        vol = metrics['Portfolio_Volatility_Percentage']
        beta = metrics['Portfolio_Beta']

        # Extra features
        div_index = 1.0 - np.sum(np.array(weights)**2)

        portfolio_sectors = {}
        port_cap_score = 0.0
        
        for t, w in zip(tickers, weights):
            score = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else 2.0
            port_cap_score += w * score
            sector = meta_df.loc[t, "Sector"] if (t in meta_df.index and "Sector" in meta_df.columns) else loader.sector_map.get(t, "Unknown")
            portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + w

        weighted_sector_vol = 0.0
        weighted_sector_beta = 0.0
        
        for sec, sec_weight in portfolio_sectors.items():
            sec_tickers = [tk for tk, s in loader.sector_map.items() if s == sec]
            s_vol, s_beta = calc.calculate_sector_metrics(sec_tickers)
            weighted_sector_vol += sec_weight * s_vol
            weighted_sector_beta += sec_weight * s_beta

        labeler = RiskLabeler()
        score_result = labeler.calculate_final_score(
            port_q_pct=vol, 
            port_b=beta, 
            sector_q=weighted_sector_vol, 
            sector_b=weighted_sector_beta
        )

        model_path = os.path.join(BASE_DIR, "models", "risk_classifier.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Prediction using 6 features
            input_df = pd.DataFrame(
                [[vol, beta, weighted_sector_vol * 100, weighted_sector_beta, div_index, port_cap_score]], 
                columns=['Portfolio_Volatility', 'Portfolio_Beta', 'Sector_Volatility', 'Sector_Beta', 'Diversification_Index', 'Market_Cap_Score']
            )
            ai_category = model.predict(input_df)[0]
            
            # Probabilities
            probs = model.predict_proba(input_df)[0]
            classes = model.classes_
            prob_str = " | ".join([f"{c}: {p*100:.0f}%" for c, p in zip(classes, probs)])
        else:
            ai_category = "Model Not Found (Train first!)"
            prob_str = "N/A"

        # Output formatting
        print("\n" + "*" * 45)
        print("            RISK ANALYSIS RESULTS")
        print("*" * 45)
        print(f"\n  {'METRIC':<30}{'VALUE':>12}")
        print(f"  {'-'*42}")
        print(f"  {'Portfolio Volatility':<30}{vol:>11}%")
        print(f"  {'Portfolio Beta':<30}{beta:>12.3f}")
        print(f"  {'Diversification Index':<30}{div_index:>12.3f}")
        print(f"  {'Market Cap Score':<30}{port_cap_score:>12.2f}")
        print(f"  {'Sector Volatility':<30}{weighted_sector_vol*100:>11.2f}%")
        print(f"  {'Sector Beta':<30}{weighted_sector_beta:>12.2f}")
        print(f"\n  {'-'*42}")
        print(f"  {'Risk Score (Math)':<30}{score_result['Final_Risk_Score']:>12}")
        print(f"  {'MATH CLASSIFICATION':<30}{score_result['Risk_Category']:>12}")
        print(f"  {'AI CLASSIFICATION':<30}{ai_category:>12}")
        print(f"  {'AI Confidence':<15} {prob_str}")

        print(f"\n  SECTOR BREAKDOWN")
        print(f"  {'-'*42}")
        for sector_name, weight in portfolio_sectors.items():
            print(f"  {sector_name:<20}{weight*100:>6.1f}%")

        print("\n" + "*" * 45)

        # Show details from labeler
        details = score_result.get('Details', {})
        if details:
            print(f"\n  Detailed Breakdown:")
            print(f"  Norm. Volatility Score: "
                  f"{details.get('Normalized_Port_Vol', 'N/A')}")
            print(f"  Norm. Beta Score:       "
                  f"{details.get('Normalized_Port_Beta', 'N/A')}")
            print(f"  Raw Sector Risk:        "
                  f"{details.get('Raw_Sector_Risk', 'N/A')}")
            print(f"  Norm. Sector Risk:      "
                  f"{details.get('Normalized_Sector_Risk', 'N/A')}")
            print()

    except Exception as e:
        print(f"\n  [Error] Could not calculate risk: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 45)
    print("  TADAWUL PORTFOLIO RISK ANALYZER")
    print("  Powered by Math + AI")
    print("=" * 45)

    while True:
        tickers, weights = get_user_portfolio()

        if tickers and len(tickers) > 0:
            process_prediction(tickers, weights)
            
        if input("Analyze another portfolio? (y/n): ").lower() != 'y':
            break