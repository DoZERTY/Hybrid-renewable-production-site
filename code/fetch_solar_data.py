import requests
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")


# Clé ENTSO-E — variable d'environnement ou fallback hardcodé
ENTSOE_KEY = os.environ.get("ENTSOE_API_KEY", "ef66c1cf-3999-4a36-b321-4676249806e3")
# Prix électricité ENTSO-E
try:
    from entsoe import EntsoePandasClient
    ENTSOE_AVAILABLE = True
except ImportError:
    ENTSOE_AVAILABLE = False
    print("⚠ entsoe-py non installé. Lancer : pip install entsoe-py")



# ===========================================================================
# 1. CONFIGURATION DU PROJET
# ===========================================================================

# ---- Coordonnées du site ----
SITE = {
    "name"      : "Leucate (Aude)",
    "latitude"  : 42.917,   # °N
    "longitude" : 3.037,    # °E — côte entre étang et mer, optimisé
    "altitude"  : 12,       # m NGF (terrain bas, hors garrigue)
    "timezone"  : "Europe/Paris",
}

# ---- Paramètres du parc ----
PARK = {
    "power_mwp"         : 20.0,    # Puissance crête installée [MWp]
    "tilt_deg"          : 25,      # Inclinaison repos tracker [°] — position par défaut (stow)
    "azimuth_deg"       : 180,     # Orientation axe tracker (180° = axe N-S, rotation E→O)
    "gcr"               : 0.4,    # Ground Coverage Ratio — tracker mono-axe (rangées espacées vs fixe 0.40)
    "module_efficiency" : 0.225,   # Rendement TOPCon bifacial Type-N — η~22.5% STC (Wang et al. 2024, Prog. Photovolt.)
    "pr_p50"            : 0.83,    # Performance Ratio P50 — tracker (IAM réduit, ombrage inter-rangées réduit)
    "pr_p90"            : 0.80,    # PR P90 conservateur — intègre pertes stow tramontane (~2%)
    "degradation_pct_yr": 0.35,    # Dégradation annuelle [%/an] — TOPCon Type-N (< PERC grâce au n-type)
    "lifetime_years"    : 30,      # Durée de vie projet
}


# ---- Période d'analyse ----
PERIOD = {
    "start" : "2015-01-01",
    "end"   : "2024-12-31",
    "years" : 10,
}


# ---- Hypothèses économiques ----
ECONOMICS = {
    "capex_eur_kwp"     : 800,    # CAPEX [€/kWp] — tracker mono-axe TOPCon France 2024 (fixe ~740, +8% tracker)
    "opex_eur_kwp_yr"   : 12,     # OPEX annuel [€/kWp/an]
    "p50_price_eur_mwh" : 65,     # Prix de vente P50 [€/MWh] — complément PPA/M0
    "p90_price_eur_mwh" : 58,     # Prix P90 (scénario conservateur)
    "discount_rate"     : 0.055,  # Taux d'actualisation (WACC)
    "land_lease_eur_ha_yr" : 2000,  
    "land_area_ha"         : 22.22,     # Surface terrain [ha] — fixe GCR 0.40
    "land_lease_indexation": 0.020,  # Indexation annuelle ICC/IPC [%/an]
}



# ===========================================================================
# 2. COLLECTE DES DONNÉES MÉTÉO — OPEN-METEO ARCHIVE
# ===========================================================================

def fetch_weather_solar(
    lat        : float = SITE["latitude"],
    lon        : float = SITE["longitude"],
    start_date : str   = PERIOD["start"],
    end_date   : str   = PERIOD["end"],
) -> pd.DataFrame:

    """
    Récupère les données météo horaires depuis Open-Meteo Archive API.

    Variables solaires retournées :
      - shortwave_radiation        : GHI [W/m²] — rayonnement global horizontal
      - diffuse_radiation          : DHI [W/m²] — rayonnement diffus horizontal
      - direct_normal_irradiance   : DNI [W/m²] — rayonnement direct normal
      - direct_radiation           : Beam Horizontal Irradiance [W/m²]
      - sunshine_duration          : Durée d'ensoleillement [s/h] → fraction solaire

    Variables météo complémentaires :
      - temperature_2m             : Température air à 2m [°C] — critique pour Tmodule
      - wind_speed_10m             : Vent à 10m [m/s] — refroidissement modules
      - relative_humidity_2m       : Humidité relative [%] — soiling, brouillard
      - cloud_cover                : Couverture nuageuse [%]
      - precipitation              : Précipitations [mm] — auto-nettoyage soiling

    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": start_date,
        "end_date"  : end_date,
        "hourly"    : [
            # === IRRADIANCE ===
            "shortwave_radiation",          # GHI [W/m²]
            "diffuse_radiation",            # DHI [W/m²]
            "direct_normal_irradiance",     # DNI [W/m²]
            "direct_radiation",             # BHI = GHI - DHI [W/m²]
            "global_tilted_irradiance",     # GTI à 25°S si dispo — vérification
            "sunshine_duration",            # [s] par heure
            # === MÉTÉO ===
            "temperature_2m",               # [°C]
            "wind_speed_10m",               # [m/s]
            "relative_humidity_2m",         # [%]
            "cloud_cover",                  # [%]
            "precipitation",                # [mm]
        ],
        "timezone": "UTC",
    }

    print(f"→ Téléchargement données météo {start_date} → {end_date}...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    # Renommage pour clarté
    df = df.rename(columns={
        "shortwave_radiation"       : "GHI",
        "diffuse_radiation"         : "DHI",
        "direct_normal_irradiance"  : "DNI",
        "direct_radiation"          : "BHI",
        "global_tilted_irradiance"  : "GTI_raw",
        "sunshine_duration"         : "sunshine_s",
        "temperature_2m"            : "temp_air",
        "wind_speed_10m"            : "wind_speed",
        "relative_humidity_2m"      : "humidity",
        "cloud_cover"               : "cloud_cover",
        "precipitation"             : "precipitation",
    })

    # Qualité : valeurs négatives → 0
    for col in ["GHI", "DHI", "DNI", "BHI"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Fraction d'ensoleillement (0–1)
    df["sunshine_frac"] = (df["sunshine_s"] / 3600).clip(0, 1)

    print(f"✓ {len(df):,} heures récupérées ({df.index[0]} → {df.index[-1]})")
    print(f"  GHI moyen annuel estimé : {df['GHI'].sum() / PERIOD['years'] / 1000:.0f} kWh/m²/an")

    return df


# ===========================================================================
# 4. COLLECTE DONNÉES PRIX ÉLECTRICITÉ
# ===========================================================================

def fetch_electricity_prices(
    api_key    : str = ENTSOE_KEY,          # utilise la constante globale
    start      : str = "2020-01-01",
    end        : str = "2024-12-31",
    country    : str = "FR",
) -> pd.DataFrame:

    if not ENTSOE_AVAILABLE:
        print("⚠ entsoe-py non installé → prix synthétiques")
        return _generate_synthetic_prices()

    if api_key is None or api_key == "":
        print("⚠ Clé ENTSO-E manquante → prix synthétiques")
        return _generate_synthetic_prices()

    try:
        client = EntsoePandasClient(api_key=api_key)

        # Requête par année pour éviter les timeouts
        dfs = []
        for year in range(int(start[:4]), int(end[:4]) + 1):
            print(f"  → Téléchargement prix {year}...")
            df_year = client.query_day_ahead_prices(
                country_code=country,
                start=pd.Timestamp(f"{year}-01-01", tz="Europe/Paris"),
                end=pd.Timestamp(f"{year}-12-31", tz="Europe/Paris"),
            )
            dfs.append(df_year)

        df_prices = pd.concat(dfs)
        df_prices = df_prices.rename("price_eur_mwh").to_frame()
        df_prices.index = df_prices.index.tz_convert("UTC").tz_localize(None)
        df_prices = df_prices.resample("h").mean()

        print(f"✓ Prix ENTSO-E récupérés : {len(df_prices):,} heures")
        print(f"  Prix moyen : {df_prices['price_eur_mwh'].mean():.1f} €/MWh")
        return df_prices

    except Exception as e:
        print(f"⚠ Erreur ENTSO-E : {e}")
        print("  → Bascule sur prix synthétiques")
        return _generate_synthetic_prices()


def _generate_synthetic_prices() -> pd.DataFrame:
    """
    Prix synthétiques réalistes basés sur profil France 2020–2024.
    Intègre saisonnalité, pic morning/evening, duck curve solaire.
    """
    idx = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h")
    np.random.seed(42)

    hours = idx.hour
    months = idx.month

    # Base saisonnière (€/MWh) — hiver cher, été moyen
    base = 70 + 25 * np.cos(2 * np.pi * (months - 1) / 12)

    # Profil journalier — duck curve solaire
    morning_peak = 20 * np.exp(-0.5 * ((hours - 8) / 2) ** 2)
    evening_peak = 25 * np.exp(-0.5 * ((hours - 19) / 2) ** 2)
    solar_valley = -15 * np.exp(-0.5 * ((hours - 13) / 3) ** 2)  # "duck curve"

    price = base + morning_peak + evening_peak + solar_valley
    price += np.random.normal(0, 8, len(idx))  # Bruit de marché
    price = np.maximum(price, -20)  # Prix négatifs possibles

    df = pd.DataFrame({"price_eur_mwh": price}, index=idx)
    print(f"✓ Prix synthétiques générés ({len(df):,} heures)")
    return df