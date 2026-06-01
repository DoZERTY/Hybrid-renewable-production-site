import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")
import os

from fetch_solar_data import SITE, PARK, PERIOD, ECONOMICS


# PV modelling — standard industrie
try:
    import pvlib
    from pvlib.location import Location
    from pvlib.irradiance import get_extra_radiation
    import pvlib.irradiance as pvl_irr
    PVLIB_AVAILABLE = True
    print(f"✓ pvlib {pvlib.__version__} disponible — modélisation physique complète activée")
except ImportError:
    PVLIB_AVAILABLE = False
    pvl_irr = None
    print(" pvlib non installé. Lancer : pip install pvlib")
    print("  Mode dégradé : production estimée par corrélation GHI directe")




# ===========================================================================
# 3. MODÉLISATION PV — pvlib
# ===========================================================================

def compute_pv_production(df_weather: pd.DataFrame) -> pd.DataFrame:
    if not PVLIB_AVAILABLE:
        # Mode dégradé : corrélation empirique GHI → production
        print("⚠ Mode dégradé — estimation production par corrélation GHI")
        df = df_weather.copy()
        # Production ≈ GHI × surface_utile × rendement_module × PR
        surface_m2 = PARK["power_mwp"] * 1e6 / (1000 * PARK["module_efficiency"])
        df["P_AC_MW"] = (df["GHI"] / 1000) * PARK["module_efficiency"] * PARK["pr_p50"] * surface_m2 / 1e6
        df["P_AC_MW"] = df["P_AC_MW"].clip(upper=PARK["power_mwp"])
        return df

    print("→ Modélisation PV avec pvlib...")

    # Localisation pvlib
    location = Location(
        latitude=SITE["latitude"],
        longitude=SITE["longitude"],
        altitude=SITE["altitude"],
        tz=SITE["timezone"],
        name=SITE["name"],
    )

    # Index en timezone locale pour pvlib
    times_local = df_weather.index.tz_localize("UTC").tz_convert(SITE["timezone"])

    # ---- 3.1 Position solaire ----
    solar_pos = location.get_solarposition(times_local)

    # ---- 3.2 Irradiance extraterrestre ----
    dni_extra = get_extra_radiation(times_local)

    # ---- 3.3 Masse d'air relative ----
    try:
        from pvlib.atmosphere import get_relative_airmass as _get_airmass
    except ImportError:
        from pvlib.irradiance import get_relative_airmass as _get_airmass
    airmass = _get_airmass(solar_pos["apparent_zenith"])

    # ---- 3.4 Décomposition GHI → DHI/DNI si non dispo ou incohérent ----
    ghi = df_weather["GHI"].values
    dhi = df_weather["DHI"].values
    dni = df_weather["DNI"].values

    # Vérification cohérence GHI ≈ DHI + DNI*cos(zenith)
    zenith_rad = np.radians(solar_pos["apparent_zenith"].values)
    ghi_check = dhi + dni * np.cos(np.clip(zenith_rad, 0, np.pi/2))
    residual = np.abs(ghi - ghi_check)
    pct_inconsistent = (residual > 50).mean() * 100
    if pct_inconsistent > 5:
        print(f"  ⚠ {pct_inconsistent:.1f}% des heures : incohérence GHI/DHI/DNI > 50 W/m²")
        print("    → Recalcul DNI/DHI via modèle Erbs (décomposition depuis GHI)")
        # pvlib 0.15.x : erbs() retourne un DataFrame (colonnes: dni, dhi, kt)
        decomp = pvl_irr.erbs(
            ghi=pd.Series(ghi, index=times_local),
            zenith=solar_pos["apparent_zenith"],
            datetime_or_doy=times_local,
        )
        if hasattr(decomp, "dni"):       # namedtuple (anciennes versions)
            dni = np.array(decomp.dni)
            dhi = np.array(decomp.dhi)
        else:                            # DataFrame (pvlib >= 0.10)
            dni = decomp["dni"].values
            dhi = decomp["dhi"].values
        dni = np.nan_to_num(dni, nan=0.0).clip(min=0)
        dhi = np.nan_to_num(dhi, nan=0.0).clip(min=0)
    else:
        print(f"  ✓ Cohérence GHI/DNI/DHI : {100 - pct_inconsistent:.1f}% des heures OK")

    # ---- 3.5 Transposition POA — pvlib.irradiance.get_total_irradiance() ----
    poa_df = pvl_irr.get_total_irradiance(
        surface_tilt=PARK["tilt_deg"],
        surface_azimuth=PARK["azimuth_deg"],
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=pd.Series(dni, index=times_local),
        ghi=pd.Series(ghi, index=times_local),
        dhi=pd.Series(dhi, index=times_local),
        dni_extra=dni_extra,
        airmass=airmass,
        model="perez",    # Ciel anisotropique Perez — IEA recommandé
        albedo=0.20,      # Albédo sol : garrigue/calcaire Leucate
    )
    # Accès garanti par nom de colonne (DataFrame)
    poa_global = poa_df["poa_global"].fillna(0).clip(lower=0)

    # ---- 3.6 Effet bifacial simplifié ----
    # Gain bifacial ≈ bifaciality × albedo × poa_rear/poa_front
    bifacial_gain = 0.090  # 9.0% gain net — TOPCon bifacialité 80-85%, tracker mono-axe GCR=0.33
                           # (IEA PVPS Task 13, 2024 : gain bifacial tracker 8-12% vs 3-5% fixe)
    poa_effective = poa_global * (1 + bifacial_gain)

    # ---- 3.7 Température de cellule (modèle SAPM — Sandia) ----
    temp_air = df_weather["temp_air"].values
    wind_speed = df_weather["wind_speed"].values

    # Modèle Ross amélioré avec effet vent
    noct = 43  # °C — NOCT TOPCon bifacial verre-verre Type-N
    k_noct = (noct - 20) / 800  # °C/(W/m²)
    k_wind = 0.03              # Coefficient correction vent [°C/(m/s)]
    temp_cell = temp_air + k_noct * poa_effective.values - k_wind * wind_speed

    # ---- 3.8 Correction de rendement en température ----
    # P_DC = P_STC × (POA/1000) × [1 + γ × (T_cell - 25)]
    gamma = -0.0032   # Coefficient de température [/°C] — TOPCon Type-N (meilleur que PERC -0.35%/°C)
    p_dc_normalized = (poa_effective / 1000) * (1 + gamma * (temp_cell - 25))
    p_dc_normalized = p_dc_normalized.clip(lower=0)

    # ---- 3.9 Production DC [MWh] ----
    p_dc_mw = p_dc_normalized * PARK["power_mwp"]

    # ---- 3.10 Pertes AC → Production nette ----
    # Valeur retenue : +20% net (conservateur, intègre pertes stow tramontane ~2%)
    TRACKER_GAIN = 1.20
    p_ac_mw = p_dc_mw * PARK["pr_p50"] * TRACKER_GAIN
    p_ac_mw = p_ac_mw.clip(upper=PARK["power_mwp"], lower=0)

    # ---- Assemblage DataFrame résultats ----
    df_out = df_weather.copy()
    df_out["poa_global"]   = poa_global.values
    df_out["poa_effective"] = poa_effective.values
    df_out["temp_cell"]    = temp_cell
    df_out["P_DC_MW"]      = p_dc_mw.values
    df_out["P_AC_MW"]      = p_ac_mw.values

    annual_prod = p_ac_mw.sum() / PERIOD["years"] / 1000  # GWh/an
    cf = p_ac_mw.mean() / PARK["power_mwp"] * 100
    print(f"✓ Production estimée P50 : {annual_prod:.1f} GWh/an")
    print(f"  Facteur de capacité     : {cf:.1f}%")
    print(f"  Équivalent heures pleine puissance : {annual_prod*1000/PARK['power_mwp']:.0f} h/an")
    surface_ha = PARK["power_mwp"] * 1000000 / (1000*PARK["module_efficiency"]) / PARK["gcr"] / 10000
    print(f"  Surface terrain estimée (GCR={PARK['gcr']}) : ~{surface_ha:.0f} ha")

    return df_out



# ===========================================================================
# 5. ANALYSE DU POTENTIEL SOLAIRE
# ===========================================================================

def analyze_solar_resource(df: pd.DataFrame) -> dict:
    """
    Calcule les KPIs clés du potentiel solaire pour le rapport de pré-faisabilité.
    Retourne un dictionnaire de métriques.
    """
    results = {}

    # ---- KPI annuels ----
    annual = df.resample("YE").agg({
        "GHI"    : "sum",
        "P_AC_MW": "sum",
    })
    annual["GHI_kWh_m2"] = annual["GHI"] / 1000
    annual["prod_GWh"]   = annual["P_AC_MW"] / 1000
    annual["cf_pct"]     = annual["P_AC_MW"] / (PARK["power_mwp"] * 8760) * 100
    annual["yield_kWh_kWp"] = annual["P_AC_MW"] * 1000 / PARK["power_mwp"]

    results["annual"] = annual

    # ---- Statistiques P50/P90 ----
    # P50 = médiane, P90 = 10ème percentile (90% de probabilité de dépasser)
    results["P50_GWh_yr"]    = float(annual["prod_GWh"].median())
    results["P90_GWh_yr"]    = float(annual["prod_GWh"].quantile(0.10))
    results["mean_GWh_yr"]   = float(annual["prod_GWh"].mean())
    results["std_GWh_yr"]    = float(annual["prod_GWh"].std())
    results["cv_pct"]        = results["std_GWh_yr"] / results["mean_GWh_yr"] * 100
    results["P50_yield"]     = float(annual["yield_kWh_kWp"].median())
    results["P50_CF_pct"]    = float(annual["cf_pct"].median())
    results["GHI_mean_kWh"]  = float(annual["GHI_kWh_m2"].mean())

    # ---- Profil mensuel moyen ----
    monthly = df.groupby(df.index.month).agg({
        "GHI"    : "mean",
        "P_AC_MW": "mean",
    })
    monthly.index.name = "month"
    monthly["prod_MWh_moy"] = monthly["P_AC_MW"] * 24 * 30.44  # Estimation mensuelle
    results["monthly"] = monthly

    # ---- Profil journalier moyen par saison ----
    seasons = {
        "Été (JJA)"     : [6, 7, 8],
        "Automne (SON)" : [9, 10, 11],
        "Hiver (DJF)"   : [12, 1, 2],
        "Printemps (MAM)": [3, 4, 5],
    }
    hourly_season = {}
    for season, months_list in seasons.items():
        mask = df.index.month.isin(months_list)
        hourly_season[season] = df[mask].groupby(df[mask].index.hour)["P_AC_MW"].mean()
    results["hourly_season"] = hourly_season

    # ---- Heures de production par tranche ----
    bins = [0, 0.1, 2, 5, 10, 15, 20]
    labels = ["Nulle", "< 2 MW", "2–5 MW", "5–10 MW", "10–15 MW", "15–20 MW"]
    prod_dist = pd.cut(df["P_AC_MW"], bins=bins, labels=labels).value_counts()
    results["production_distribution"] = prod_dist

    return results



# ===========================================================================
# 6. ANALYSE ÉCONOMIQUE PRÉLIMINAIRE
# ===========================================================================

def compute_economics(df: pd.DataFrame, df_prices: pd.DataFrame, kpi: dict) -> dict:
    """
    Calcule les indicateurs économiques de base pour la pré-faisabilité.
    Hypothèses simplifiées — à affiner lors des études de faisabilité complète.
    """
    eco = {}

    # ---- CAPEX ----
    power_kwp = PARK["power_mwp"] * 1000
    eco["capex_meur"]  = ECONOMICS["capex_eur_kwp"] * power_kwp / 1e6
    eco["opex_meur_yr"]= ECONOMICS["opex_eur_kwp_yr"] * power_kwp / 1e6

    # ---- Revenus annuels ----
    # Merge production et prix
    df_merged = df[["P_AC_MW"]].copy()
    df_merged = df_merged.join(df_prices, how="left")
    df_merged["price_eur_mwh"] = df_merged["price_eur_mwh"].fillna(
        ECONOMICS["p50_price_eur_mwh"]
    )

    # Revenue = somme(P_AC [MW] * Price [€/MWh] * 1h)
    df_merged["revenue_eur_h"] = df_merged["P_AC_MW"] * df_merged["price_eur_mwh"]
    annual_rev = df_merged.resample("YE")["revenue_eur_h"].sum()
    eco["revenue_meur_yr_p50"] = float(annual_rev.median() / 1e6)
    eco["revenue_meur_yr_p90"] = float(annual_rev.quantile(0.10) / 1e6)

    # ---- Capture rate (prix moyen obtenu vs prix de marché moyen) ----
    prod_mask = df_merged["P_AC_MW"] > 0.1
    eco["capture_rate_pct"] = (
        df_merged.loc[prod_mask, "price_eur_mwh"].mean() /
        df_merged["price_eur_mwh"].mean() * 100
    )

    # ---- EBITDA et indicateurs simples ----
    eco["ebitda_meur_yr"] = eco["revenue_meur_yr_p50"] - eco["opex_meur_yr"]

    # ---- Durée de retour simple ----
    eco["payback_simple_yr"] = eco["capex_meur"] / eco["ebitda_meur_yr"]

    # ---- LCOE simplifié ----
    # LCOE = (CAPEX + VAN(OPEX)) / VAN(Production)
    r = ECONOMICS["discount_rate"]
    n = PARK["lifetime_years"]
    annuity_factor = (1 - (1 + r) ** (-n)) / r

    prod_annual_gwh = kpi["P50_GWh_yr"]
    # Production dégradée sur durée de vie
    years = np.arange(1, n + 1)
    deg = (1 - PARK["degradation_pct_yr"] / 100) ** years
    prod_lifetime_discounted = sum(
        prod_annual_gwh * d / (1 + r) ** y
        for y, d in zip(years, deg)
    )
    opex_discounted = eco["opex_meur_yr"] * annuity_factor

    eco["lcoe_eur_mwh"] = (eco["capex_meur"] + opex_discounted) / prod_lifetime_discounted * 1000
    eco["prod_lifetime_gwh"] = prod_annual_gwh * sum(deg)

    # ---- NPV simplifié P50 ----
    cashflows = [(eco["ebitda_meur_yr"] * d) / (1 + r) ** y
                 for y, d in zip(years, deg)]
    eco["npv_meur_p50"] = sum(cashflows) - eco["capex_meur"]

    # ---- LCOE cible marché ----
    eco["target_ppa_eur_mwh"] = eco["lcoe_eur_mwh"] * 1.15  # Marge 15% pour viabilité

    # Loyer foncier annuel (indexé)
    land_lease_yr0 = ECONOMICS["land_lease_eur_ha_yr"] * ECONOMICS["land_area_ha"] / 1e6  # M€
    # Loyer cumulé actualisé sur 30 ans
    land_lease_npv = sum(
        land_lease_yr0 * (1 + ECONOMICS["land_lease_indexation"]) ** y / (1 + r) ** y
        for y in range(1, n + 1)
    )
    eco["land_lease_meur_yr"]  = land_lease_yr0
    eco["land_lease_npv_meur"] = land_lease_npv

    return eco