import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os

from fetch_wind_data import SITE, PARK, PERIOD, ECONOMICS, power_curve_v236


# ===========================================================================
# 3. MODELISATION EOLIENNE
# ===========================================================================

def compute_wind_production(df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit le vent horaire en production nette du parc (MW) via la courbe V236.

    Etapes :
      1. Puissance brute d'une turbine via courbe V236
      2. x nombre de turbines = puissance brute parc
      3. x pertes (sillage, cables, transfo, disponibilite) = puissance nette
    """
    print("-> Modelisation eolienne (courbe V236-15.0 MW)...")
    df = df_weather.copy()

    # 1. Puissance brute d'une turbine (MW)
    p_turbine = power_curve_v236(df["wind_speed"].values)

    # 2. Puissance brute parc (MW)
    p_park_gross = p_turbine * PARK["n_turbines"]

    # 3. Pertes systeme cumulees
    pertes = (
        PARK["wake_loss"]
        * PARK["loss_cables_inter"]
        * PARK["loss_cable_export"]
        * PARK["loss_transfo"]
        * PARK["availability"]
    )
    p_park_net = p_park_gross * pertes
    # Plafonnement a la puissance installee
    p_park_net = np.clip(p_park_net, 0, PARK["power_mw"])

    # ---- Assemblage : on nomme la colonne P_AC_MW pour COMPATIBILITE avec
    # la chaine solaire (visualisation et economie reutilisent ce nom) ----
    df["P_turbine_MW"]   = p_turbine
    df["P_gross_MW"]     = p_park_gross
    df["P_AC_MW"]        = p_park_net   # production nette parc (MW)

    annual_prod = p_park_net.sum() / PERIOD["years"] / 1000  # GWh/an
    cf = p_park_net.mean() / PARK["power_mw"] * 100
    print(f"OK Production estimee P50 : {annual_prod:.1f} GWh/an")
    print(f"  Pertes systeme totales  : {(1 - pertes) * 100:.1f}%")
    print(f"  Facteur de charge       : {cf:.1f}%")
    print(f"  Heures equivalentes     : {annual_prod * 1000 / PARK['power_mw']:.0f} h/an")

    return df


# ===========================================================================
# 5. ANALYSE DE LA RESSOURCE EOLIENNE
# ===========================================================================

def analyze_wind_resource(df: pd.DataFrame) -> dict:
    """
    KPIs cles du potentiel eolien. Structure IDENTIQUE a analyze_solar_resource
    pour que visualisation_wind et l'economie hybride fonctionnent sans heurt.
    """
    results = {}

    # ---- KPI annuels ----
    annual = df.resample("YE").agg({
        "wind_speed": "mean",
        "P_AC_MW"   : "sum",
    })
    annual["wind_mean_ms"] = annual["wind_speed"]
    annual["prod_GWh"]     = annual["P_AC_MW"] / 1000
    annual["cf_pct"]       = annual["P_AC_MW"] / (PARK["power_mw"] * 8760) * 100
    # Yield = energie annuelle (MWh/an) / puissance installee (kW) -> kWh/kW/an
    # annual["P_AC_MW"] est la SOMME horaire => deja en MWh/an.
    # MWh/an * 1000 (kWh) / (power_mw * 1000) (kW)  =  P_AC_MW.sum() / power_mw
    annual["yield_kWh_kW"] = annual["P_AC_MW"] / PARK["power_mw"]
    results["annual"] = annual

    # ---- Statistiques P50/P90 ----
    results["P50_GWh_yr"]  = float(annual["prod_GWh"].median())
    results["P90_GWh_yr"]  = float(annual["prod_GWh"].quantile(0.10))
    results["mean_GWh_yr"] = float(annual["prod_GWh"].mean())
    results["std_GWh_yr"]  = float(annual["prod_GWh"].std())
    results["cv_pct"]      = results["std_GWh_yr"] / results["mean_GWh_yr"] * 100
    results["P50_yield"]   = float(annual["yield_kWh_kW"].median())
    results["P50_CF_pct"]  = float(annual["cf_pct"].median())
    results["wind_mean_ms"] = float(annual["wind_mean_ms"].mean())

    # ---- Profil mensuel moyen ----
    monthly = df.groupby(df.index.month).agg({
        "wind_speed": "mean",
        "P_AC_MW"   : "mean",
    })
    monthly.index.name = "month"
    monthly["prod_MWh_moy"] = monthly["P_AC_MW"] * 24 * 30.44
    results["monthly"] = monthly

    # ---- Profil journalier moyen par saison ----
    seasons = {
        "Ete (JJA)"      : [6, 7, 8],
        "Automne (SON)"  : [9, 10, 11],
        "Hiver (DJF)"    : [12, 1, 2],
        "Printemps (MAM)": [3, 4, 5],
    }
    hourly_season = {}
    for season, months_list in seasons.items():
        mask = df.index.month.isin(months_list)
        hourly_season[season] = df[mask].groupby(df[mask].index.hour)["P_AC_MW"].mean()
    results["hourly_season"] = hourly_season

    # ---- Distribution de production par tranche ----
    pmax = PARK["power_mw"]
    bins = [0, 0.1, 0.25 * pmax, 0.5 * pmax, 0.75 * pmax, pmax]
    labels = ["Nulle", "< 25%", "25-50%", "50-75%", "75-100%"]
    prod_dist = pd.cut(df["P_AC_MW"], bins=bins, labels=labels).value_counts()
    results["production_distribution"] = prod_dist

    # ---- Rose des vents (bonus, donnees reelles) ----
    if "wind_direction" in df.columns:
        dir_bins = np.arange(0, 361, 30)
        results["wind_rose"] = pd.cut(
            df["wind_direction"], bins=dir_bins
        ).value_counts().sort_index()

    return results


# ===========================================================================
# 6. ANALYSE ECONOMIQUE PRELIMINAIRE (EOLIEN SEUL)
# ===========================================================================

def compute_economics(df: pd.DataFrame, df_prices: pd.DataFrame, kpi: dict) -> dict:
    """
    Indicateurs economiques eolien seul. Memes cles de sortie que la version solaire.
    """
    eco = {}

    power_kw = PARK["power_mw"] * 1000
    eco["capex_meur"]   = ECONOMICS["capex_eur_kw"] * power_kw / 1e6
    eco["opex_meur_yr"] = ECONOMICS["opex_eur_kw_yr"] * power_kw / 1e6

    # ---- Revenus au prix horaire REEL capte (ENTSO-E day-ahead) ----
    df_merged = df[["P_AC_MW"]].copy()
    df_merged = df_merged.join(df_prices, how="left")
    df_merged["price_eur_mwh"] = df_merged["price_eur_mwh"].fillna(
        ECONOMICS["p50_price_eur_mwh"]
    )
    df_merged["revenue_eur_h"] = df_merged["P_AC_MW"] * df_merged["price_eur_mwh"]
    annual_rev = df_merged.resample("YE")["revenue_eur_h"].sum()
    eco["revenue_meur_yr_p50"] = float(annual_rev.median() / 1e6)
    eco["revenue_meur_yr_p90"] = float(annual_rev.quantile(0.10) / 1e6)

    # ---- Prix de capture REEL = revenu total / energie totale (EUR/MWh) ----
    # C'est le prix moyen effectivement obtenu, pondere par la production horaire.
    energy_total_mwh = df_merged["P_AC_MW"].sum()
    revenue_total_eur = df_merged["revenue_eur_h"].sum()
    eco["capture_price_eur_mwh"] = (
        revenue_total_eur / energy_total_mwh if energy_total_mwh > 0 else 0.0
    )
    # Prix marche moyen (baseload, non pondere)
    eco["baseload_price_eur_mwh"] = float(df_merged["price_eur_mwh"].mean())
    # Ratio de capture = prix capte / baseload (en %)
    eco["capture_rate_pct"] = (
        eco["capture_price_eur_mwh"] / eco["baseload_price_eur_mwh"] * 100
        if eco["baseload_price_eur_mwh"] else 0.0
    )

    eco["ebitda_meur_yr"]    = eco["revenue_meur_yr_p50"] - eco["opex_meur_yr"]
    eco["payback_simple_yr"] = (
        eco["capex_meur"] / eco["ebitda_meur_yr"]
        if eco["ebitda_meur_yr"] > 0 else float("inf")
    )

    # ---- LCOE actualise ----
    r = ECONOMICS["discount_rate"]
    n = PARK["lifetime_years"]
    annuity_factor = (1 - (1 + r) ** (-n)) / r
    prod_annual_gwh = kpi["P50_GWh_yr"]
    years = np.arange(1, n + 1)
    deg = (1 - PARK["degradation_pct_yr"] / 100) ** years
    prod_lifetime_disc = sum(
        prod_annual_gwh * d / (1 + r) ** y for y, d in zip(years, deg)
    )
    opex_disc = eco["opex_meur_yr"] * annuity_factor
    eco["lcoe_eur_mwh"]      = (eco["capex_meur"] + opex_disc) / prod_lifetime_disc * 1000
    eco["prod_lifetime_gwh"] = prod_annual_gwh * sum(deg)

    # ---- MARGE REELLE : prix capte - LCOE (EUR/MWh) ----
    # Indicateur clé : reconcilie NPV et LCOE. Si > 0, le projet cree de la valeur
    # AU PRIX REELLEMENT CAPTE (et non a un prix de vente theorique fixe).
    eco["margin_capture_vs_lcoe"] = (
        eco["capture_price_eur_mwh"] - eco["lcoe_eur_mwh"]
    )

    # ---- NPV sur revenu REEL capte (coherente avec le prix de capture) ----
    cashflows = [(eco["ebitda_meur_yr"] * d) / (1 + r) ** y
                 for y, d in zip(years, deg)]
    eco["npv_meur_p50"]      = sum(cashflows) - eco["capex_meur"]
    eco["target_ppa_eur_mwh"] = eco["lcoe_eur_mwh"] * 1.15

    return eco
