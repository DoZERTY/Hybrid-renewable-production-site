import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fetch_wind_data import SITE, PARK, ECONOMICS


def plot_wind_resource(df: pd.DataFrame, kpi: dict, eco: dict):
    """Figures cles du rapport de pre-faisabilite eolien (miroir du solaire)."""

    # =========================================================
    # FIGURE 1 — VUE D'ENSEMBLE DU POTENTIEL EOLIEN
    # =========================================================
    fig1, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig1.suptitle(
        f"Analyse du Potentiel Eolien - {SITE['name']}\n"
        f"Parc {PARK['power_mw']:.0f} MW | {PARK['n_turbines']} x {PARK['turbine_model']}",
        fontsize=12
    )

    annual = kpi["annual"]
    years = annual.index.year

    # --- 1.1 Vent moyen annuel ---
    ax = axes[0, 0]
    ax.bar(years, annual["wind_mean_ms"], color="tab:cyan")
    ax.axhline(annual["wind_mean_ms"].mean(), color="tab:red", linestyle="--",
               label=f"Moyenne: {annual['wind_mean_ms'].mean():.2f} m/s")
    ax.set_title("Vent Moyen Annuel a 100m (m/s)")
    ax.set_xlabel("Annee")
    ax.legend(fontsize=8)

    # --- 1.2 Production annuelle P50/P90 ---
    ax = axes[0, 1]
    ax.bar(years, annual["prod_GWh"], color="tab:blue")
    p50, p90 = kpi["P50_GWh_yr"], kpi["P90_GWh_yr"]
    ax.axhline(p50, color="tab:orange", linestyle="-", label=f"P50: {p50:.1f}")
    ax.axhline(p90, color="tab:red", linestyle="--", label=f"P90: {p90:.1f}")
    ax.set_title("Production Annuelle Nette (GWh/an)")
    ax.legend(fontsize=8)

    # --- 1.3 Profil mensuel moyen ---
    ax = axes[0, 2]
    monthly = kpi["monthly"]
    month_names = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(range(1, 13), monthly["wind_speed"], color="tab:cyan", alpha=0.6,
           label="Vent (m/s)")
    ax2_twin = ax.twinx()
    ax2_twin.plot(range(1, 13), monthly["P_AC_MW"], color="tab:blue",
                  marker="o", label="Puissance (MW)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45)
    ax.set_title("Profil Mensuel Moyen")
    ax.set_ylabel("Vent (m/s)")
    ax2_twin.set_ylabel("Puissance (MW)")
    # Axes a ZERO pour une comparaison honnete (evite d'amplifier de faibles ecarts)
    ax.set_ylim(0, max(monthly["wind_speed"]) * 1.15)
    ax2_twin.set_ylim(0, max(monthly["P_AC_MW"]) * 1.15)
    ax.legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)

    # --- 1.4 Profil journalier saisonnier ---
    ax = axes[1, 0]
    ymax = 0
    for season, series in kpi["hourly_season"].items():
        ax.plot(series.index, series.values, label=season, marker=".")
        ymax = max(ymax, series.max())
    ax.set_title("Profil Journalier Moyen par Saison")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Puissance (MW)")
    # Axe a ZERO : un parc offshore a peu de cycle jour/nuit, l'echelle zoomee
    # exagererait des variations faibles. On montre l'amplitude reelle.
    ax.set_ylim(0, ymax * 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 1.5 Courbe de duree de production ---
    ax = axes[1, 1]
    production_sorted = np.sort(df["P_AC_MW"].values)[::-1]
    hours_axis = np.arange(1, len(production_sorted) + 1) / len(production_sorted) * 8760
    ax.plot(hours_axis, production_sorted, color="tab:blue")
    ax.fill_between(hours_axis, production_sorted, alpha=0.2, color="tab:blue")
    ax.axhline(PARK["power_mw"] * 0.5, color="tab:red", linestyle=":",
               label="50% Pnom")
    ax.set_title("Courbe de Duree de Production")
    ax.set_xlabel("Heures/an")
    ax.set_ylabel("Puissance (MW)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 1.6 Tableau des KPIs ---
    ax = axes[1, 2]
    ax.axis("off")
    kpi_table = [
        ["Indicateur", "Valeur", "Unite"],
        ["Vent moyen 100m", f"{kpi['wind_mean_ms']:.2f}", "m/s"],
        ["Production P50", f"{kpi['P50_GWh_yr']:.1f}", "GWh/an"],
        ["Yield P50", f"{kpi['P50_yield']:.0f}", "kWh/kW/an"],
        ["Facteur de charge", f"{kpi['P50_CF_pct']:.1f}", "%"],
        ["LCOE estime", f"{eco['lcoe_eur_mwh']:.1f}", "EUR/MWh"],
        ["Prix capte reel", f"{eco['capture_price_eur_mwh']:.1f}", "EUR/MWh"],
        ["Ratio de capture", f"{eco['capture_rate_pct']:.0f}", "%"],
        ["Marge (capte-LCOE)", f"{eco['margin_capture_vs_lcoe']:+.1f}", "EUR/MWh"],
        ["CAPEX", f"{eco['capex_meur']:.0f}", "M EUR"],
        ["Revenu P50", f"{eco['revenue_meur_yr_p50']:.1f}", "M EUR/an"],
        ["NPV (P50, 30 ans)", f"{eco['npv_meur_p50']:.0f}", "M EUR"],
    ]
    table = ax.table(cellText=kpi_table[1:], colLabels=kpi_table[0],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.2)
    ax.set_title("Synthese KPIs")

    plt.tight_layout()
    plt.show()

    # =========================================================
    # FIGURE 2 — ANALYSE ECONOMIQUE
    # =========================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Analyse Economique Preliminaire - Eolien", fontsize=12)

    # --- 2.1 Sensibilite LCOE vs CAPEX ---
    ax = axes2[0]
    capex_range = np.linspace(2800, 4100, 50)  # EUR/kW offshore flottant
    r, n = ECONOMICS["discount_rate"], PARK["lifetime_years"]
    annuity = (1 - (1 + r) ** (-n)) / r
    opex_discounted = eco["opex_meur_yr"] * annuity
    lcoe_range = []
    for c_kw in capex_range:
        capex_m = c_kw * PARK["power_mw"] * 1000 / 1e6
        lcoe = (capex_m + opex_discounted) / eco["prod_lifetime_gwh"] * 1000
        lcoe_range.append(lcoe)
    ax.plot(capex_range, lcoe_range, color="tab:blue")
    ax.axvline(ECONOMICS["capex_eur_kw"], color="tab:red", linestyle="--",
               label=f"CAPEX retenu ({ECONOMICS['capex_eur_kw']})")
    ax.axhline(eco["capture_price_eur_mwh"], color="tab:green", linestyle=":",
               label=f"Prix capte reel ({eco['capture_price_eur_mwh']:.0f})")
    ax.set_title("Sensibilite LCOE vs CAPEX")
    ax.set_xlabel("CAPEX (EUR/kW)")
    ax.set_ylabel("LCOE (EUR/MWh)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 2.2 NPV cumule actualise ---
    ax = axes2[1]
    years_cf = np.arange(1, PARK["lifetime_years"] + 1)
    degradation = (1 - PARK["degradation_pct_yr"] / 100) ** years_cf
    ebitda_series = eco["ebitda_meur_yr"] * degradation
    ebitda_discounted = ebitda_series / (1 + r) ** years_cf
    cumulative_npv = np.cumsum(ebitda_discounted) - eco["capex_meur"]
    ax.plot(years_cf, cumulative_npv, color="tab:blue", label="NPV cumule")
    ax.axhline(0, color="black", linewidth=0.8)
    payback_idx = np.where(cumulative_npv >= 0)[0]
    if len(payback_idx) > 0:
        pb_yr = years_cf[payback_idx[0]]
        ax.axvline(pb_yr, color="tab:red", linestyle="--",
                   label=f"Payback: {pb_yr} ans")
    ax.set_title("NPV Cumule Actualise (30 ans)")
    ax.set_xlabel("Annees")
    ax.set_ylabel("NPV (M EUR)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
