import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fetch_solar_data import SITE, PARK, ECONOMICS



def plot_solar_resource(df: pd.DataFrame, kpi: dict, eco: dict):
    """
    Génère les figures clés du rapport de pré-faisabilité solaire (Style standard).
    """

    # =========================================================
    # FIGURE 1 — VUE D'ENSEMBLE DU POTENTIEL SOLAIRE
    # =========================================================
    fig1, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig1.suptitle(
        f"Analyse du Potentiel Solaire - {SITE['name']}\n"
        f"Parc {PARK['power_mwp']} MWp | Tracker Mono-Axe | TOPCon Bifacial | GCR={PARK['gcr']}",
        fontsize=12
    )

    # --- 1.1 GHI annuel ---
    ax = axes[0, 0]
    annual = kpi["annual"]
    years = annual.index.year
    bars = ax.bar(years, annual["GHI_kWh_m2"], color="tab:blue")
    ax.axhline(annual["GHI_kWh_m2"].mean(), color="tab:red", linestyle="--", 
               label=f"Moyenne: {annual['GHI_kWh_m2'].mean():.0f}")
    ax.set_title("GHI Annuel (kWh/m²/an)")
    ax.set_xlabel("Année")
    ax.legend(fontsize=8)

    # --- 1.2 Production annuelle P50/P90 ---
    ax = axes[0, 1]
    ax.bar(years, annual["prod_GWh"], color="tab:green")
    p50 = kpi["P50_GWh_yr"]
    p90 = kpi["P90_GWh_yr"]
    ax.axhline(p50, color="tab:orange", linestyle="-", label=f"P50: {p50:.1f}")
    ax.axhline(p90, color="tab:red", linestyle="--", label=f"P90: {p90:.1f}")
    ax.set_title("Production Annuelle Nette (GWh/an)")
    ax.legend(fontsize=8)

    # --- 1.3 Profil mensuel moyen ---
    ax = axes[0, 2]
    monthly = kpi["monthly"]
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    ax.bar(range(1, 13), monthly["GHI"] / 1000, color="tab:blue", alpha=0.6, label="GHI (kW/m²)")
    ax2_twin = ax.twinx()
    ax2_twin.plot(range(1, 13), monthly["P_AC_MW"], color="tab:green", marker="o", label="Puissance (MW)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_title("Profil Mensuel Moyen")
    ax.set_ylabel("GHI (kW/m²)")
    ax2_twin.set_ylabel("Puissance AC (MW)")
    ax.legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)

    # --- 1.4 Profil journalier saisonnier ---
    ax = axes[1, 0]
    for season, series in kpi["hourly_season"].items():
        ax.plot(series.index, series.values, label=season, marker=".")
    ax.set_title("Profil Journalier Moyen par Saison")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Puissance AC (MW)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 1.5 Distribution de production (courbe de charge) ---
    ax = axes[1, 1]
    production_sorted = np.sort(df["P_AC_MW"].values)[::-1]
    hours_total = len(production_sorted)
    hours_axis = np.arange(1, hours_total + 1) / hours_total * 8760 

    ax.plot(hours_axis, production_sorted, color="tab:green")
    ax.fill_between(hours_axis, production_sorted, alpha=0.2, color="tab:green")
    ax.axhline(PARK["power_mwp"] * 0.5, color="tab:red", linestyle=":", label="50% Pnom")
    ax.set_title("Courbe de Durée de Production")
    ax.set_xlabel("Heures/an")
    ax.set_ylabel("Puissance AC (MW)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 1.6 Tableau des KPIs ---
    ax = axes[1, 2]
    ax.axis("off")

    kpi_table = [
        ["Indicateur", "Valeur", "Unité"],
        ["GHI moyen annuel", f"{kpi['GHI_mean_kWh']:.0f}", "kWh/m²/an"],
        ["Production P50", f"{kpi['P50_GWh_yr']:.1f}", "GWh/an"],
        ["Production P90", f"{kpi['P90_GWh_yr']:.1f}", "GWh/an"],
        ["Yield P50", f"{kpi['P50_yield']:.0f}", "kWh/kWp/an"],
        ["Facteur de capacité", f"{kpi['P50_CF_pct']:.1f}", "%"],
        ["Variabilité (CoV)", f"{kpi['cv_pct']:.1f}", "%"],
        ["LCOE estimé", f"{eco['lcoe_eur_mwh']:.1f}", "€/MWh"],
        ["CAPEX", f"{eco['capex_meur']:.1f}", "M€"],
        ["Revenu P50", f"{eco['revenue_meur_yr_p50']:.2f}", "M€/an"],
        ["NPV (P50, 30 ans)", f"{eco['npv_meur_p50']:.1f}", "M€"],
    ]

    table = ax.table(
        cellText=kpi_table[1:],
        colLabels=kpi_table[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.2)
    ax.set_title("Synthèse KPIs")

    plt.tight_layout()
    plt.show()

    # =========================================================
    # FIGURE 2 — ANALYSE ÉCONOMIQUE
    # =========================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Analyse Économique Préliminaire", fontsize=12)

    # --- 2.1 Sensibilité LCOE vs CAPEX ---
    ax = axes2[0]
    capex_range = np.linspace(550, 950, 50)
    lcoe_range = []
    r, n = ECONOMICS["discount_rate"], PARK["lifetime_years"]
    annuity = (1 - (1 + r) ** (-n)) / r
    opex_discounted = eco["opex_meur_yr"] * annuity
    
    for c_kwp in capex_range:
        capex_m = c_kwp * PARK["power_mwp"] * 1000 / 1e6
        lcoe = (capex_m + opex_discounted) / (eco["prod_lifetime_gwh"]) * 1000
        lcoe_range.append(lcoe)

    ax.plot(capex_range, lcoe_range, color="tab:blue")
    ax.axvline(ECONOMICS["capex_eur_kwp"], color="tab:red", linestyle="--", label=f"CAPEX retenu ({ECONOMICS['capex_eur_kwp']})")
    ax.axhline(ECONOMICS["p50_price_eur_mwh"], color="tab:green", linestyle=":", label=f"Prix cible ({ECONOMICS['p50_price_eur_mwh']})")
    ax.set_title("Sensibilité LCOE vs CAPEX")
    ax.set_xlabel("CAPEX (€/kWp)")
    ax.set_ylabel("LCOE (€/MWh)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 2.2 Cash-flows actualisés ---
    ax = axes2[1]
    years_cf = np.arange(1, PARK["lifetime_years"] + 1)
    degradation = (1 - PARK["degradation_pct_yr"] / 100) ** years_cf
    ebitda_series = eco["ebitda_meur_yr"] * degradation
    ebitda_discounted = ebitda_series / (1 + r) ** years_cf
    cumulative_npv = np.cumsum(ebitda_discounted) - eco["capex_meur"]

    ax.plot(years_cf, cumulative_npv, color="tab:blue", label="NPV cumulé")
    ax.axhline(0, color="black", linewidth=0.8)

    payback_idx = np.where(cumulative_npv >= 0)[0]
    if len(payback_idx) > 0:
        pb_yr = years_cf[payback_idx[0]]
        ax.axvline(pb_yr, color="tab:red", linestyle="--", label=f"Payback: {pb_yr} ans")

    ax.set_title("NPV Cumulé Actualisé (30 ans)")
    ax.set_xlabel("Années")
    ax.set_ylabel("NPV (M€)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()