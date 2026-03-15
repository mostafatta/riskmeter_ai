import sys
import os

# === Fix imports to work from any directory ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from calculations import RiskCalculator
from risk_labeler import RiskLabeler
from data_loader import TadawulDataLoader

def generate_dataset(num_samples=500):
    print(f"--- GENERATING {num_samples} RANDOM PORTFOLIOS ---\n")

    # ==========================================
    # 1. Load Data from Tadawul
    # ==========================================
    loader = TadawulDataLoader()
    loader.fetch_stock_data()
    loader.fetch_market_data()
    loader.fetch_metadata()

    calc = RiskCalculator()
    calc.load_data()
    calc.calculate_daily_returns()

    labeler = RiskLabeler()

    # ==========================================
    # 2. Load Metadata (Market Cap + Sector)
    # ==========================================
    meta_path = os.path.join(loader.data_dir, "stocks_metadata.csv")
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path).set_index("Ticker")
    else:
        meta_df = None

    dataset = []
    available_tickers = calc.tickers

    for i in range(num_samples):
        try:
            num_stocks = np.random.randint(1, 8)
            selected_tickers = np.random.choice(available_tickers, num_stocks, replace=False)
            weights = np.random.dirichlet(np.ones(num_stocks), size=1)[0]

            full_weights = [0.0] * len(available_tickers)
            port_cap_score = 0.0
            portfolio_sectors = {}

            for idx, ticker in enumerate(selected_tickers):
                full_idx = available_tickers.index(ticker)
                full_weights[full_idx] = weights[idx]
                
                if meta_df is not None and ticker in meta_df.index:
                    score = meta_df.loc[ticker, "Market_Cap_Score"]
                    sector = meta_df.loc[ticker, "Sector"]
                else:
                    score = 2.0
                    sector = loader.sector_map.get(ticker, "Unknown")

                port_cap_score += weights[idx] * score
                portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + weights[idx]

            div_index = 1.0 - np.sum(np.array(full_weights) ** 2)
            metrics = calc.calculate_portfolio_risk(full_weights)

            weighted_sector_vol = 0.0
            weighted_sector_beta = 0.0

            for sec, sec_weight in portfolio_sectors.items():
                sec_tickers = [tk for tk, s in loader.sector_map.items() if s == sec]
                s_vol, s_beta = calc.calculate_sector_metrics(sec_tickers)
                weighted_sector_vol += sec_weight * s_vol
                weighted_sector_beta += sec_weight * s_beta

            # === 🔥 حقن حالات High Risk 🔥 ===
            vol_pct = metrics['Portfolio_Volatility_Percentage']
            beta = metrics['Portfolio_Beta']
            
            # في 20% من الحالات، سنجبر الأرقام لتكون خطيرة جداً
            if np.random.rand() > 0.8:
                vol_pct = np.random.uniform(28.0, 40.0)       # تذبذب عنيف
                beta = np.random.uniform(1.2, 1.8)            # حساسية مفرطة للسوق
                weighted_sector_vol = np.random.uniform(0.28, 0.40) 
                weighted_sector_beta = np.random.uniform(1.2, 1.6)
                port_cap_score = np.random.uniform(1.0, 1.5)  # شركات صغيرة ومخاطرة
                div_index = np.random.uniform(0.0, 0.2)       # تنويع سيء جداً

            result = labeler.calculate_final_score(
                port_q_pct=vol_pct,
                port_b=beta,
                sector_q=weighted_sector_vol,
                sector_b=weighted_sector_beta
            )

            dataset.append({
                "Portfolio_Volatility": round(vol_pct, 2),
                "Portfolio_Beta": round(beta, 3),
                "Sector_Volatility": round(weighted_sector_vol * 100, 2),
                "Sector_Beta": round(weighted_sector_beta, 2),
                "Diversification_Index": round(div_index, 3),
                "Market_Cap_Score": round(port_cap_score, 2),
                "Risk_Score": result['Final_Risk_Score'],
                "Risk_Category": result['Risk_Category'],
            })

        except Exception as e:
            continue

    df = pd.DataFrame(dataset)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'portfolio_dataset.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"  Generated {len(df)} portfolios successfully!")
    print(f"{'='*50}")
    print("\n🔥 Risk Category Distribution:")
    print(df['Risk_Category'].value_counts())
    return df

if __name__ == "__main__":
    generate_dataset(500)