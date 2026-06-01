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

    # # =========================================================
    # # FIGURE 2 — ANALYSE ÉCONOMIQUE
    # # =========================================================
    # fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    # fig2.suptitle("Analyse Économique Préliminaire", fontsize=12)

    # # --- 2.1 Sensibilité LCOE vs CAPEX ---
    # ax = axes2[0]
    # capex_range = np.linspace(550, 950, 50)
    # lcoe_range = []
    # r, n = ECONOMICS["discount_rate"], PARK["lifetime_years"]
    # annuity = (1 - (1 + r) ** (-n)) / r
    # opex_discounted = eco["opex_meur_yr"] * annuity
    
    # for c_kwp in capex_range:
    #     capex_m = c_kwp * PARK["power_mwp"] * 1000 / 1e6
    #     lcoe = (capex_m + opex_discounted) / (eco["prod_lifetime_gwh"]) * 1000
    #     lcoe_range.append(lcoe)

    # ax.plot(capex_range, lcoe_range, color="tab:blue")
    # ax.axvline(ECONOMICS["capex_eur_kwp"], color="tab:red", linestyle="--", label=f"CAPEX retenu ({ECONOMICS['capex_eur_kwp']})")
    # ax.axhline(ECONOMICS["p50_price_eur_mwh"], color="tab:green", linestyle=":", label=f"Prix cible ({ECONOMICS['p50_price_eur_mwh']})")
    # ax.set_title("Sensibilité LCOE vs CAPEX")
    # ax.set_xlabel("CAPEX (€/kWp)")
    # ax.set_ylabel("LCOE (€/MWh)")
    # ax.legend(fontsize=8)
    # ax.grid(True, linestyle=":", alpha=0.6)

    # # --- 2.2 Cash-flows actualisés ---
    # ax = axes2[1]
    # years_cf = np.arange(1, PARK["lifetime_years"] + 1)
    # degradation = (1 - PARK["degradation_pct_yr"] / 100) ** years_cf
    # ebitda_series = eco["ebitda_meur_yr"] * degradation
    # ebitda_discounted = ebitda_series / (1 + r) ** years_cf
    # cumulative_npv = np.cumsum(ebitda_discounted) - eco["capex_meur"]

    # ax.plot(years_cf, cumulative_npv, color="tab:blue", label="NPV cumulé")
    # ax.axhline(0, color="black", linewidth=0.8)

    # payback_idx = np.where(cumulative_npv >= 0)[0]
    # if len(payback_idx) > 0:
    #     pb_yr = years_cf[payback_idx[0]]
    #     ax.axvline(pb_yr, color="tab:red", linestyle="--", label=f"Payback: {pb_yr} ans")

    # ax.set_title("NPV Cumulé Actualisé (30 ans)")
    # ax.set_xlabel("Années")
    # ax.set_ylabel("NPV (M€)")
    # ax.legend(fontsize=8)
    # ax.grid(True, linestyle=":", alpha=0.6)

    # plt.tight_layout()
    # plt.show()

    # =========================================================
# FIGURE 2 — ANALYSE ÉCONOMIQUE
# =========================================================
#
# CORRECTIONS LCOE vs code original :
#
#   Formule LCOE =  (CAPEX + VAN(OPEX) + VAN(Loyer))  [€]
#                   ─────────────────────────────────────
#                        VAN(Production)               [MWh]
#
#   Unités homogènes imposées PARTOUT :
#     - CAPEX          : M€  → converti en €  (× 1e6)
#     - OPEX actualisé : M€  → converti en €  (× 1e6)
#     - Loyer actualisé: M€  → converti en €  (× 1e6)
#     - Production     : GWh → converti en MWh (× 1e3)
#     - Résultat LCOE  : €/MWh  (pas besoin de × 1000 final)
#
#   Bug original : capex_m et opex_discounted étaient en M€,
#   prod_lifetime_gwh en GWh, et le ×1000 ne compensait pas
#   correctement → LCOE sous-estimé d'un facteur ~1000/1e6 = ×0.001
#   soit environ ×0.001×1000 = sous-estimation d'un facteur ~1.
#   En pratique le ×1000 donnait M€/GWh = k€/MWh ≠ €/MWh → ×1000 manquant.
#   Résultat observé : 36.7 €/MWh au lieu de ~50 €/MWh.

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Analyse Économique Préliminaire — 20 MWp Leucate", fontsize=12, fontweight="bold")

    r = ECONOMICS["discount_rate"]
    n = PARK["lifetime_years"]

    # ── Facteur d'actualisation (annuité) ─────────────────────────────────────
    annuity_factor = (1 - (1 + r) ** (-n)) / r

    # ── OPEX actualisé [€] — inclut loyer foncier ─────────────────────────────
    # On sépare OPEX technique et loyer pour la traçabilité
    opex_tech_eur_yr  = eco["opex_meur_yr"] * 1e6          # €/an
    land_lease_eur_yr = eco["land_lease_meur_yr"] * 1e6    # €/an

    # Le loyer est indexé (croissant) → actualisation exacte année par année
    years_arr = np.arange(1, n + 1)
    land_lease_pv = sum(
        land_lease_eur_yr * (1 + ECONOMICS["land_lease_indexation"]) ** y / (1 + r) ** y
        for y in years_arr
    )
    opex_tech_pv = opex_tech_eur_yr * annuity_factor        # OPEX constant → annuité simple

    # ── Production actualisée sur durée de vie [MWh] ──────────────────────────
    # Dégradation annuelle composée + actualisation temporelle
    prod_p50_mwh_yr = kpi["P50_GWh_yr"] * 1e3             # GWh → MWh
    degradation_arr = (1 - PARK["degradation_pct_yr"] / 100) ** years_arr

    prod_lifetime_discounted_mwh = sum(
        prod_p50_mwh_yr * deg / (1 + r) ** y
        for y, deg in zip(years_arr, degradation_arr)
    )

    # ── Production non-actualisée (pour courbe NPV) ───────────────────────────
    prod_lifetime_mwh = prod_p50_mwh_yr * sum(degradation_arr)


    # =========================================================
    # 2.1  Sensibilité LCOE vs CAPEX
    # =========================================================
    ax = axes2[0]
    capex_range_kwp = np.linspace(550, 1050, 100)   # €/kWp
    lcoe_range       = []

    for c_kwp in capex_range_kwp:
        # CAPEX total [€]
        capex_eur = c_kwp * PARK["power_mwp"] * 1e3   # kWp × €/kWp = €

        # Numérateur [€] : CAPEX + OPEX actualisé + Loyer actualisé
        numerator_eur = capex_eur + opex_tech_pv + land_lease_pv

        # LCOE [€/MWh] — unités cohérentes
        lcoe = numerator_eur / prod_lifetime_discounted_mwh
        lcoe_range.append(lcoe)

    lcoe_range = np.array(lcoe_range)

    # LCOE au CAPEX retenu (ligne de référence)
    capex_ref_eur = ECONOMICS["capex_eur_kwp"] * PARK["power_mwp"] * 1e3
    lcoe_ref = (capex_ref_eur + opex_tech_pv + land_lease_pv) / prod_lifetime_discounted_mwh

    ax.plot(capex_range_kwp, lcoe_range, color="tab:blue", lw=2)
    ax.axvline(
        ECONOMICS["capex_eur_kwp"], color="tab:red", linestyle="--", lw=1.5,
        label=f"CAPEX retenu : {ECONOMICS['capex_eur_kwp']} €/kWp\nLCOE = {lcoe_ref:.1f} €/MWh"
    )
    ax.axhline(
        ECONOMICS["p50_price_eur_mwh"], color="tab:green", linestyle=":", lw=1.5,
        label=f"Prix PPA cible : {ECONOMICS['p50_price_eur_mwh']} €/MWh"
    )
    # Zone de viabilité (LCOE < prix PPA)
    ax.fill_between(
        capex_range_kwp, lcoe_range, ECONOMICS["p50_price_eur_mwh"],
        where=(lcoe_range < ECONOMICS["p50_price_eur_mwh"]),
        alpha=0.15, color="green", label="Zone viable (P50)"
    )

    ax.set_title("Sensibilité LCOE vs CAPEX", fontsize=10)
    ax.set_xlabel("CAPEX (€/kWp)")
    ax.set_ylabel("LCOE (€/MWh)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(capex_range_kwp[0], capex_range_kwp[-1])


    # =========================================================
    # 2.2  Cash-flows actualisés et NPV cumulé
    # =========================================================
    ax = axes2[1]

    # Revenue annuel P50 avec dégradation [M€/an]
    revenue_yr = eco["revenue_meur_yr_p50"] * degradation_arr

    # OPEX total (technique + loyer, indexé) [M€/an]
    opex_total_yr = (
        eco["opex_meur_yr"]
        + eco["land_lease_meur_yr"] * (1 + ECONOMICS["land_lease_indexation"]) ** years_arr
    )

    # EBITDA annuel [M€/an]
    ebitda_yr = revenue_yr - opex_total_yr

    # Cash-flow actualisé [M€]
    cf_discounted = ebitda_yr / (1 + r) ** years_arr

    # NPV cumulé [M€] — on soustrait le CAPEX initial à t=0
    cumulative_npv = np.cumsum(cf_discounted) - eco["capex_meur"]

    ax.bar(
        years_arr, cf_discounted,
        color=np.where(cf_discounted >= 0, "#2a9d8f", "#e63946"),
        alpha=0.6, label="Cash-flow annuel actualisé"
    )
    ax.plot(years_arr, cumulative_npv, color="tab:blue", lw=2, label="NPV cumulé")
    ax.axhline(0, color="black", linewidth=0.8)

    # Payback
    payback_idx = np.where(cumulative_npv >= 0)[0]
    if len(payback_idx) > 0:
        pb_yr = years_arr[payback_idx[0]]
        ax.axvline(
            pb_yr, color="tab:red", linestyle="--", lw=1.5,
            label=f"Payback actualisé : {pb_yr} ans"
        )
        

    # NPV finale
    npv_final = cumulative_npv[-1]
    ax.annotate(
        f"NPV 30 ans\n{npv_final:.1f} M€",
        xy=(n, npv_final),
        xytext=(n - 8, npv_final * 0.8),
        fontsize=8, color="tab:blue",
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
    )

    ax.set_title("NPV Cumulé Actualisé (30 ans)", fontsize=10)
    ax.set_xlabel("Années")
    ax.set_ylabel("M€")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0, n + 1)

    plt.tight_layout()
    plt.savefig("figure2_economics.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Récapitulatif console ──────────────────────────────────────────────────
    print("\n── LCOE corrigé ─────────────────────────────────────────────────────")
    print(f"  CAPEX total            : {capex_ref_eur/1e6:.1f} M€")
    print(f"  OPEX tech. actualisé   : {opex_tech_pv/1e6:.1f} M€")
    print(f"  Loyer foncier actualisé: {land_lease_pv/1e6:.2f} M€")
    print(f"  Production actualisée  : {prod_lifetime_discounted_mwh/1e3:.0f} GWh")
    print(f"  LCOE                   : {lcoe_ref:.1f} €/MWh")
    print(f"  NPV P50 (30 ans)       : {npv_final:.1f} M€")
    print(f"  Payback actualisé      : {pb_yr if len(payback_idx) > 0 else 'N/A'} ans")
    print("─────────────────────────────────────────────────────────────────────")