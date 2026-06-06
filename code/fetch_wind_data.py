import requests
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")


# Cle ENTSO-E — UNIQUEMENT via variable d'environnement (ne JAMAIS hardcoder).
# Sur le portail ENTSO-E : regenerer la cle si elle a deja ete exposee dans du code.
#   export ENTSOE_API_KEY="votre-cle"
ENTSOE_KEY = os.environ.get("ENTSOE_API_KEY", "")

# Prix electricite ENTSO-E
try:
    from entsoe import EntsoePandasClient
    ENTSOE_AVAILABLE = True
except ImportError:
    ENTSOE_AVAILABLE = False
    print("entsoe-py non installe. Lancer : pip install entsoe-py")


# ===========================================================================
# 1. CONFIGURATION DU PROJET — PARC EOLIEN OFFSHORE FLOTTANT
# ===========================================================================

# ---- Coordonnees du site (parc offshore, identique aux scripts ERA5) ----
SITE = {
    "name"      : "Leucate Offshore (Golfe du Lion)",
    "latitude"  : 42.95,    # N
    "longitude" : 3.07,     # E — cellule maritime offshore
    "altitude"  : 0,        # niveau de la mer
    "timezone"  : "Europe/Paris",
}

# ---- Parametres du parc eolien (Vestas V236-15.0 MW) ----
PARK = {
    "turbine_model"     : "Vestas V236-15.0 MW",
    "n_turbines"        : 17,      # 17 x 15 MW = 255 MW
    "power_per_turbine" : 15.0,    # MW
    "power_mw"          : 255.0,   # Puissance installee totale [MW] (17 x 15)
    "hub_height_m"      : 150,     # Hauteur moyeu [m] (vent Open-Meteo a 100 m, voir extrapolation)
    "rotor_diameter_m"  : 236,     # m
    "swept_area_m2"     : 43743,   # m2
    "cut_in"            : 3.0,     # m/s
    "cut_out"           : 31.0,    # m/s (V236)
    "availability"      : 0.95,    # Disponibilite globale offshore flottant
    "degradation_pct_yr": 0.30,    # Degradation annuelle [%/an] — turbines modernes (IEA 2024)
    "lifetime_years"    : 30,      # Duree de vie projet (V236 concue pour 30 ans)
    # Pertes systeme (hors disponibilite, deja dans 'availability') :
    #   cables inter 1.5% + cable export 2.5% + transformation 1.5%
    "loss_cables_inter" : 0.985,
    "loss_cable_export" : 0.975,
    "loss_transfo"      : 0.985,
    "wake_loss"         : 0.92,    # Pertes de sillage (~8%) — grandes turbines espacees
    # Option extrapolation vent 100 m -> hauteur moyeu (loi log)
    "extrapolate_hub"   : False,   # False = prudent (reste a 100 m)
    "roughness_sea_z0"  : 0.0002,  # m
}

# ---- Periode d'analyse ----
PERIOD = {
    "start" : "2015-01-01",
    "end"   : "2024-12-31",
    "years" : 10,
}

# ---- Hypotheses economiques (offshore flottant V236, coherentes avec l'Excel) ----
ECONOMICS = {
    "capex_eur_kw"      : 3350,    # CAPEX [EUR/kW] — V236 (2050 turbines+flotteurs + 480 cables + 820 install)
    "opex_eur_kw_yr"    : 60,      # OPEX annuel [EUR/kW/an] — commercial a l'echelle 2024-25
    "p50_price_eur_mwh" : 95,      # Prix de vente P50 [EUR/MWh] — AO commercial prudent
    "p90_price_eur_mwh" : 80,      # Prix P90 conservateur
    "discount_rate"     : 0.0474,  # WACC (coherent avec le modele Excel)
    "tax_rate"          : 0.25,    # IS France
}


# ===========================================================================
# 2. COURBE DE PUISSANCE V236-15.0 MW
# ===========================================================================
# Courbe approchee a partir des specs publiques (rotor 236 m, 43 743 m2,
# 15 MW, cut-in 3, cut-out 31, puissance specifique 343 W/m2).
# ATTENTION : remplacer par la courbe certifiee Vestas pour un dossier bancable.

COURBE_V236 = {
    0.0:     0,   0.5:     0,   1.0:     0,   1.5:     0,   2.0:     0,
    2.5:     0,
    3.0:   300,   3.5:   600,   4.0:  1050,   4.5:  1600,   5.0:  2350,
    5.5:  3250,   6.0:  4350,   6.5:  5600,   7.0:  7050,   7.5:  8650,
    8.0: 10400,   8.5: 12100,   9.0: 13500,   9.5: 14400,  10.0: 14850,
   10.5: 14980,  11.0: 15000,  11.5: 15000,  12.0: 15000,  12.5: 15000,
   13.0: 15000,  13.5: 15000,  14.0: 15000,  14.5: 15000,  15.0: 15000,
   16.0: 15000,  17.0: 15000,  18.0: 15000,  19.0: 15000,  20.0: 15000,
   21.0: 15000,  22.0: 15000,  23.0: 15000,  24.0: 15000,  25.0: 15000,
   26.0: 15000,  27.0: 15000,  28.0: 15000,  29.0: 15000,  30.0: 15000,
   31.0: 15000,
   31.5:     0,
}


def power_curve_v236(v_array: np.ndarray) -> np.ndarray:
    """Vitesse de vent (m/s) -> puissance d'UNE turbine (MW) via courbe V236."""
    vt = np.array(sorted(COURBE_V236.keys()))
    pt = np.array([COURBE_V236[v] for v in vt])
    v = np.asarray(v_array, dtype=float)
    p_kw = np.interp(v, vt, pt, left=0, right=0)
    p_kw = np.where(v >= PARK["cut_out"], 0.0, p_kw)
    p_kw = np.where(v < PARK["cut_in"], 0.0, p_kw)
    return p_kw / 1000.0  # MW


# ===========================================================================
# 3. COLLECTE DES DONNEES METEO — OPEN-METEO ARCHIVE (ERA5)
# ===========================================================================

def fetch_weather_wind(
    lat        : float = SITE["latitude"],
    lon        : float = SITE["longitude"],
    start_date : str   = PERIOD["start"],
    end_date   : str   = PERIOD["end"],
) -> pd.DataFrame:
    """
    Recupere les donnees de vent horaires depuis Open-Meteo Archive (ERA5).

    Variables retournees :
      - wind_speed_100m      : vitesse du vent a 100 m [m/s]
      - wind_direction_100m  : direction du vent a 100 m [deg]
      - temperature_2m       : temperature air [C] (densite de l'air)
      - wind_gusts_10m        : rafales [m/s] (info, episodes de cut-out)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": start_date,
        "end_date"  : end_date,
        "hourly"    : [
            "wind_speed_100m",
            "wind_direction_100m",
            "temperature_2m",
            "wind_gusts_10m",
        ],
        "wind_speed_unit": "ms",
        "timezone"  : "UTC",
        "models"    : "era5",
        "cell_selection": "sea",
    }

    print(f"-> Telechargement donnees vent {start_date} -> {end_date}...")
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    df = df.rename(columns={
        "wind_speed_100m"     : "wind_speed",
        "wind_direction_100m" : "wind_direction",
        "temperature_2m"      : "temp_air",
        "wind_gusts_10m"      : "wind_gust",
    })

    # Qualite : interpolation des NaN eventuels
    n_nan = df["wind_speed"].isna().sum()
    if n_nan > 0:
        print(f"  {n_nan} NaN detectes, interpolation...")
        df = df.interpolate()

    # Option extrapolation 100 m -> hauteur moyeu (loi log)
    if PARK["extrapolate_hub"]:
        z0 = PARK["roughness_sea_z0"]
        factor = np.log(PARK["hub_height_m"] / z0) / np.log(100.0 / z0)
        df["wind_speed"] = df["wind_speed"] * factor
        print(f"  Vent extrapole 100m -> {PARK['hub_height_m']}m (x{factor:.3f})")

    print(f"OK {len(df):,} heures recuperees ({df.index[0]} -> {df.index[-1]})")
    print(f"  Vent moyen a 100m : {df['wind_speed'].mean():.2f} m/s")
    print(f"  Vent max          : {df['wind_speed'].max():.2f} m/s")

    return df


# ===========================================================================
# 4. COLLECTE DONNEES PRIX ELECTRICITE — ENTSO-E DAY-AHEAD (FR)
# ===========================================================================

def fetch_electricity_prices(
    api_key : str = ENTSOE_KEY,
    start   : str = "2020-01-01",
    end     : str = "2024-12-31",
    country : str = "FR",
) -> pd.DataFrame:
    """Prix day-ahead reels ENTSO-E (FR). Fallback synthetique si indisponible."""
    if not ENTSOE_AVAILABLE:
        print("entsoe-py non installe -> prix synthetiques")
        return _generate_synthetic_prices(start, end)

    if not api_key:
        print("Cle ENTSO-E manquante (env ENTSOE_API_KEY) -> prix synthetiques")
        return _generate_synthetic_prices(start, end)

    try:
        client = EntsoePandasClient(api_key=api_key)
        dfs = []
        for year in range(int(start[:4]), int(end[:4]) + 1):
            print(f"  -> Telechargement prix {year}...")
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

        print(f"OK Prix ENTSO-E : {len(df_prices):,} heures")
        print(f"  Prix moyen : {df_prices['price_eur_mwh'].mean():.1f} EUR/MWh")
        return df_prices

    except Exception as e:
        print(f"Erreur ENTSO-E : {e}")
        print("  -> Bascule sur prix synthetiques")
        return _generate_synthetic_prices(start, end)


def _generate_synthetic_prices(start="2020-01-01", end="2024-12-31") -> pd.DataFrame:
    """Prix synthetiques realistes (profil France) : saisonnalite + duck curve."""
    idx = pd.date_range(start, f"{end} 23:00", freq="h")
    np.random.seed(42)
    hours = idx.hour
    months = idx.month
    base = 70 + 25 * np.cos(2 * np.pi * (months - 1) / 12)
    morning_peak = 20 * np.exp(-0.5 * ((hours - 8) / 2) ** 2)
    evening_peak = 25 * np.exp(-0.5 * ((hours - 19) / 2) ** 2)
    solar_valley = -15 * np.exp(-0.5 * ((hours - 13) / 3) ** 2)
    price = base + morning_peak + evening_peak + solar_valley
    price += np.random.normal(0, 8, len(idx))
    price = np.maximum(price, -20)
    df = pd.DataFrame({"price_eur_mwh": price}, index=idx)
    print(f"OK Prix synthetiques generes ({len(df):,} heures)")
    return df
