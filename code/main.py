"""
================================================================================
MAIN HYBRIDE - Parc Eolien Offshore (V236) + Solaire - Leucate
================================================================================

Chaine complete :
  1. Collecte vent (Open-Meteo ERA5) + production eolienne V236
  2. Collecte solaire (Open-Meteo) + production PV
  3. Collecte prix day-ahead reels ENTSO-E (FR)
  4. Croisement hybride horaire
  5. Prevision ML (production + prix) sur l'historique -> futur
  6. Valorisation au prix reel : capture, revenu, LCOE hybride, prix jour/jour
  7. Visualisations

PREREQUIS :
  pip install pandas numpy scikit-learn matplotlib requests pvlib entsoe-py
  export ENTSOE_API_KEY="votre-cle"   (sinon prix synthetiques)

NB : les modules solaires (fetch_solar_data, compute_solar_production) doivent
etre presents dans le meme dossier (vos fichiers existants, inchanges).
================================================================================
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Chargement automatique de la cle API depuis un fichier .env
# ---------------------------------------------------------------------------
# Mettez votre cle dans un fichier nomme ".env" place A COTE de ce main.py,
# avec une seule ligne (sans guillemets) :
#     ENTSOE_API_KEY=votre-cle-ici
# Le code la chargera tout seul a chaque lancement. La cle n'est JAMAIS ecrite
# dans le code => quand vous partagez vos scripts, excluez juste le fichier .env.

import pandas as pd
import numpy as np

#  EOLIEN 
from fetch_wind_data import (
    fetch_weather_wind, fetch_electricity_prices,
    SITE as WIND_SITE, PARK as WIND_PARK, PERIOD, ECONOMICS as WIND_ECO,
)

from compute_wind_production import (
    compute_wind_production, analyze_wind_resource,
    compute_economics as compute_wind_economics,
)
from visualisation_wind import plot_wind_resource

# SOLAIRE 
from fetch_solar_data import fetch_weather_solar
from compute_solar_production import (
    compute_pv_production, analyze_solar_resource,
    compute_economics as compute_solar_economics,
)
from visualisation_solar import plot_solar_resource

#  HYBRIDE + ML     
import hybrid_forecast as hf
from visualisation_hybrid import plot_hybrid


def main(show_individual=True, forecast_years=5,
         CFD_STRIKE=90.0, PPA_PRICE=80.0):
    print("=" * 72)
    print("  PARC HYBRIDE LEUCATE - EOLIEN V236 + SOLAIRE")
    print("=" * 72)

    # 
    # 1. PRIX DAY-AHEAD REELS ENTSO-E (FR) 
   
    print("\n[1/6] Prix day-ahead ENTSO-E (FR)...")
    prices_df = fetch_electricity_prices(
        start=PERIOD["start"], end=PERIOD["end"], country="FR"
    )
    print("DIAGNOSTIC prices_df brut -> min:", prices_df["price_eur_mwh"].min(),
      "| heures < 0:", (prices_df["price_eur_mwh"] < 0).sum(),
      "| total:", len(prices_df))



    # -----------------------------------------------------------------
    # 2. EOLIEN
    # -----------------------------------------------------------------
    print("\n[2/6] Eolien offshore V236...")
    wind_weather = fetch_weather_wind()
    df_wind = compute_wind_production(wind_weather)
    kpi_wind = analyze_wind_resource(df_wind)
    eco_wind = compute_wind_economics(df_wind, prices_df, kpi_wind)
    print(f"  LCOE eolien : {eco_wind['lcoe_eur_mwh']:.1f} EUR/MWh")
    from analyse_prix_negatifs import analyze_negative_prices, plot_negative_prices
    analyze_negative_prices(df_wind, prices_df)
    plot_negative_prices(prices_df)
    # -----------------------------------------------------------------
    # 3. SOLAIRE 
    # -----------------------------------------------------------------
    print("\n[3/6] Solaire PV...")
    solar_weather = fetch_weather_solar()
    df_solar = compute_pv_production(solar_weather)
    kpi_solar = analyze_solar_resource(df_solar)
    eco_solar = compute_solar_economics(df_solar, prices_df, kpi_solar)
    print(f"  LCOE solaire : {eco_solar['lcoe_eur_mwh']:.1f} EUR/MWh")

    # -----------------------------------------------------------------
    # 4. CROISEMENT HYBRIDE
    # -----------------------------------------------------------------
    print("\n[4/6] Croisement hybride...")
    hybrid = hf.build_hybrid_dataframe(df_wind, df_solar, prices_df)

    val = hf.valorize_hybrid(hybrid)
    lcoe_h = hf.hybrid_lcoe(
        eco_wind, eco_solar, kpi_wind, kpi_solar,
        discount_rate=WIND_ECO["discount_rate"],
        lifetime=WIND_PARK["lifetime_years"],
    )

    print(f"  Energie hybride   : {val['energy_hybrid_GWh_yr']:.0f} GWh/an")
    print(f"  Mix wind/solar    : {val['energy_wind_GWh_yr']:.0f} / "
          f"{val['energy_solar_GWh_yr']:.0f} GWh/an")
    print(f"  Baseload moyen    : {val['baseload_price']:.1f} EUR/MWh")
    print(f"  Capture hybride   : {val['capture_price_hybrid']:.1f} EUR/MWh "
          f"({val['capture_ratio_hybrid']:.0f}%)")
    print(f"  Capture eolien    : {val['capture_price_wind']:.1f} EUR/MWh "
          f"({val['capture_ratio_wind']:.0f}%)")
    print(f"  Capture solaire   : {val['capture_price_solar']:.1f} EUR/MWh "
          f"({val['capture_ratio_solar']:.0f}%)")
    print(f"  LCOE HYBRIDE      : {lcoe_h['lcoe_hybrid_eur_mwh']:.1f} EUR/MWh")
    print(f"  CAPEX total       : {lcoe_h['capex_total_meur']:.0f} M EUR")

    # -----------------------------------------------------------------
    # 5. SIMULATION 30 ANS : production (bootstrap) + prix (forme ML x RTE)
    #    + valorisation comparee Spot / CfD / PPA sur 3 scenarios RTE
    # -----------------------------------------------------------------
    print("\n[5/6] Simulation 30 ans (scenarios RTE + CfD/PPA)...")
    import simulation_30ans as sim
    from visualisation_30ans import plot_simulation_30y

    # CAPEX/OPEX hybrides (somme des deux parcs) repris du croisement
    capex_h = lcoe_h["capex_total_meur"]
    opex_h = lcoe_h["opex_total_meur_yr"]

    # Modeles ML : production (forme) + prix (forme, prix plafonnes)
    prod_model = sim.train_production_model(
        df_wind, power_col="P_AC_MW", weather_cols=["wind_speed"]
    )
    price_shape_model = sim.train_price_shape_model(prices_df)

    # Production de reference pour le bootstrap : hybride historique reel.
    # On reconstruit une serie horaire hybride (wind+solar) sur l'historique.
    hist_hybrid = pd.DataFrame(
        {"P_AC_MW": hybrid["P_hybrid_MW"]}, index=hybrid.index
    )

    sims, valos = {}, {}
    for scen in ["bas", "central", "haut"]:
        s = sim.simulate_30y(
            prod_model, price_shape_model, scenario=scen,
            df_prod_hist=hist_hybrid, power_col="P_AC_MW",
            start_year=2025, n_years=30,
            degradation_pct_yr=WIND_PARK["degradation_pct_yr"],
        )
        res = sim.valorize_contracts(
            s, cfd_strike=CFD_STRIKE, ppa_price=PPA_PRICE,
            discount_rate=WIND_ECO["discount_rate"],
            capex_meur=capex_h, opex_meur_yr=opex_h, power_col="P_MW",
        )
        sims[scen], valos[scen] = s, res
        print(f"\n  --- Scenario {scen.upper()} ---")
        sim.print_comparison(res)

    # -----------------------------------------------------------------
    # 6. VISUALISATIONS
    # -----------------------------------------------------------------
    print("\n[6/6] Visualisations...")
    if show_individual:
        plot_wind_resource(df_wind, kpi_wind, eco_wind)
        plot_solar_resource(df_solar, kpi_solar, eco_solar)
    plot_hybrid(hybrid, val, lcoe_h, forecast=None)
    plot_simulation_30y(sims, valos)

    print("\n" + "=" * 72)
    print("  TERMINE")
    print("=" * 72)

    return {
        "hybrid": hybrid, "val": val, "lcoe_hybrid": lcoe_h,
        "sims": sims, "valorisations": valos,
        "kpi_wind": kpi_wind, "kpi_solar": kpi_solar,
        "eco_wind": eco_wind, "eco_solar": eco_solar,
    }


if __name__ == "__main__":
    results = main(show_individual=True, forecast_years=5)


