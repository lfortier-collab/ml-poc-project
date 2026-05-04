"""
scraping.py — Enrichissement du dataset Online Shoppers Purchasing Intention
=============================================================================
Sources scrapées :
  1. Calcul algorithmique + Wikipedia REST API (validation)
     — Black Friday (4ème vendredi de novembre)
     — Cyber Monday (lundi suivant le Black Friday)
     — Cyber Week (semaine entière Black Friday -> Cyber Monday)
  2. Librairie `holidays` (locale, pas de réseau)
     — Jours fériés US uniquement (corrélation validée : 0.149)
     — Jours fériés FR supprimés (corrélation ~0.013, site non français)
  3. Amazon Prime Day — dates historiques vérifiées (juillet)

Pourquoi US uniquement ?
  L'analyse EDA montre que nb_holidays_us (corr=0.149) explique bien le
  comportement d'achat, contrairement a nb_holidays_fr (corr=0.013) et
  is_soldes (corr=-0.024). Le site source est probablement americain ou
  international, pas francais. Les soldes francaises ont ete supprimees.

Usage :
    python src/scraping.py

Resultat :
    data/enriched/commercial_events.csv        -- table mois -> features
    data/enriched/online_shoppers_enriched.csv -- dataset final enrichi

Dependances :
    pip install requests pandas holidays
"""

import requests
import pandas as pd
import holidays
from datetime import date, timedelta
import os

DATASET_PATH = "data/online_shoppers_intention.csv"
OUTPUT_DIR   = "data/enriched"
HEADERS      = {"User-Agent": "Mozilla/5.0 (compatible; student-project/1.0)"}
YEARS        = [2017, 2018]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def scrape_black_friday_dates(years):
    print("[1/3] Calcul dates Black Friday / Cyber Monday / Cyber Week...")
    results = {}
    for year in years:
        nov_1             = date(year, 11, 1)
        days_to_friday    = (4 - nov_1.weekday()) % 7
        first_friday      = nov_1 + timedelta(days=days_to_friday)
        black_friday      = first_friday + timedelta(weeks=3)
        cyber_monday      = black_friday + timedelta(days=3)
        results[year] = {
            "black_friday":      black_friday,
            "cyber_monday":      cyber_monday,
            "cyber_week_start":  black_friday,
            "cyber_week_end":    cyber_monday,
        }
        print(f"  {year} -> Black Friday: {black_friday} | Cyber Monday: {cyber_monday}")

    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/Black_Friday_(shopping)"
        r = requests.get(url, headers=HEADERS, timeout=8)
        print("  Validation Wikipedia:", "OK" if r.status_code == 200 else f"inaccessible ({r.status_code}) -- dates calculees localement")
    except Exception as e:
        print(f"  Wikipedia inaccessible ({e}) -- dates calculees localement")

    return results


def get_us_holidays(years):
    print("[2/3] Extraction jours feries US...")
    all_holidays = {}
    for year in years:
        us = holidays.UnitedStates(years=year)
        all_holidays[year] = {str(d): name for d, name in us.items()}
        nov_h = {str(d): n for d, n in us.items() if d.month == 11}
        print(f"  {year} -> {len(us)} jours feries US | Novembre: {nov_h}")
    return all_holidays


def get_prime_day_dates(years):
    """
    Amazon Prime Day historique (juillet sauf 2020=octobre).
    Juillet 2017 (15.3% conversion) > moyenne annuelle (10-11%).
    """
    print("[3/3] Chargement dates Amazon Prime Day...")
    prime_months = {2017: 7, 2018: 7, 2020: 10}
    results = {}
    for year in years:
        month = prime_months.get(year, 7)
        results[year] = {"prime_day_month": month}
        print(f"  {year} -> Prime Day mois {month}")
    return results


MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "June": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def build_monthly_features(bf_dates, us_holidays, prime_day, months, reference_year=2017):
    """
    Features retenues (toutes US-centric, corrélations validées) :
      is_black_friday_month  corr=+0.155
      is_cyber_monday_month  corr=+0.155
      is_cyber_week          corr=+0.155 (meme mois que BF)
      is_prime_day_month     corr a valider (juillet taux 15.3%)
      nb_holidays_us         corr=+0.149
      days_before_christmas  corr=-0.128
      is_pre_christmas       corr=+0.113
      is_back_to_school      corr=+0.022
      is_holiday_shopping    nouveau (oct-dec, saison cadeaux US)
      commercial_intensity   score composite 0-6

    Features supprimees :
      is_soldes_hiver / is_soldes_ete / is_soldes  corr=-0.047 a -0.006
      nb_holidays_fr                                corr=+0.013
    """
    rows = []
    for month_str in months:
        month_num = MONTH_MAP[month_str]
        yr  = reference_year
        row = {"Month": month_str}

        bf = bf_dates[yr]["black_friday"]
        cm = bf_dates[yr]["cyber_monday"]
        row["is_black_friday_month"] = int(bf.month == month_num)
        row["is_cyber_monday_month"] = int(cm.month == month_num)
        row["is_cyber_week"]         = int(bf.month == month_num)

        pd_month = prime_day[yr]["prime_day_month"]
        row["is_prime_day_month"] = int(month_num == pd_month)

        nb_us = sum(1 for d_str in us_holidays[yr]
                    if date.fromisoformat(d_str).month == month_num)
        row["nb_holidays_us"] = nb_us

        mid_month = date(yr, month_num, 15)
        christmas = date(yr, 12, 25)
        row["days_before_christmas"] = abs((christmas - mid_month).days)
        row["is_pre_christmas"]      = int(month_num in [11, 12])
        row["is_back_to_school"]     = int(month_num in [8, 9])
        row["is_holiday_shopping"]   = int(month_num in [10, 11, 12])

        row["commercial_intensity"] = (
            row["is_black_friday_month"]
            + row["is_cyber_monday_month"]
            + row["is_prime_day_month"]
            + row["is_pre_christmas"]
            + row["is_back_to_school"]
            + min(nb_us // 2, 1)
        )
        rows.append(row)

    return pd.DataFrame(rows)


def enrich_dataset(dataset_path, monthly_features, output_path):
    print("\n[Fusion] Chargement du dataset principal...")
    df = pd.read_csv(dataset_path)
    print(f"  Original  : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    df_enriched = df.merge(monthly_features, on="Month", how="left")
    new_cols = [c for c in df_enriched.columns if c not in df.columns]
    print(f"  Enrichi   : {df_enriched.shape[0]} lignes x {df_enriched.shape[1]} colonnes")
    print(f"  Nouvelles features ({len(new_cols)}) : {new_cols}")
    df_enriched.to_csv(output_path, index=False)
    print(f"  Sauvegarde -> {output_path}")
    return df_enriched, new_cols


def validate_features(df_enriched, new_cols):
    print("\n── Validation : correlations avec Revenue ──")
    df_enriched["Revenue_int"] = df_enriched["Revenue"].astype(int)
    corr = df_enriched[new_cols + ["Revenue_int"]].corr()["Revenue_int"].drop("Revenue_int")
    for feat in corr.abs().sort_values(ascending=False).index:
        val  = corr[feat]
        bar  = "+" * int(abs(val) * 50)
        sign = "+" if val >= 0 else "-"
        print(f"  {feat:<25} {sign}{abs(val):.3f}  {bar}")


def main():
    print("=" * 62)
    print("  Scraping & enrichissement -- Online Shoppers (US-centric)")
    print("=" * 62)

    bf_dates    = scrape_black_friday_dates(YEARS)
    us_hols     = get_us_holidays(YEARS)
    prime_day   = get_prime_day_dates(YEARS)

    months = ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_df = build_monthly_features(bf_dates, us_hols, prime_day, months)

    ref_path = os.path.join(OUTPUT_DIR, "commercial_events.csv")
    monthly_df.to_csv(ref_path, index=False)
    print(f"\nTable de reference -> {ref_path}")
    print(monthly_df.to_string(index=False))

    enriched_path = os.path.join(OUTPUT_DIR, "online_shoppers_enriched.csv")
    df_enriched, new_cols = enrich_dataset(DATASET_PATH, monthly_df, enriched_path)

    validate_features(df_enriched, new_cols)

    print(f"\nEnrichissement termine -- {df_enriched.shape[1]} colonnes au total")
    print("Features supprimees : is_soldes_hiver, is_soldes_ete, is_soldes, nb_holidays_fr")
    print("Features ajoutees   : is_cyber_week, is_prime_day_month, is_holiday_shopping")


if __name__ == "__main__":
    main()
