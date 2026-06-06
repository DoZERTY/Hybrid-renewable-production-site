"""
================================================================================
HYBRID FORECAST - Croisement Eolien + Solaire + Prevision ML + Valorisation
================================================================================

Ce module :
  1. Croise les productions horaires eolien + solaire (parc hybride)
  2. Entraine des modeles ML sur l'historique (production & prix day-ahead)
  3. Projette la production et le prix futurs
  4. Valorise le parc hybride au prix horaire reel (capture, revenu)
  5. Calcule le LCOE du systeme hybride et le prix capte jour par jour

ML : scikit-learn (GradientBoosting / RandomForest). Pas de dependance lourde.
Les features sont calendaires + meteo, robustes pour une prevision horaire.
================================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ===========================================================================
# 1. CROISEMENT DES PRODUCTIONS HYBRIDES
# ===========================================================================

def build_hybrid_dataframe(df_wind: pd.DataFrame,
                           df_solar: pd.DataFrame,
                           df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Aligne eolien, solaire et prix sur un index horaire commun (intersection).
    Retourne un DataFrame avec :
      - P_wind_MW, P_solar_MW, P_hybrid_MW
      - price_eur_mwh
      - features meteo conservees (wind_speed, GHI, temp_air...)
    """
    wind = df_wind[["P_AC_MW"]].rename(columns={"P_AC_MW": "P_wind_MW"}).copy()
    if "wind_speed" in df_wind.columns:
        wind["wind_speed"] = df_wind["wind_speed"]

    solar = df_solar[["P_AC_MW"]].rename(columns={"P_AC_MW": "P_solar_MW"}).copy()
    for c in ["GHI", "temp_air"]:
        if c in df_solar.columns:
            solar[c] = df_solar[c]

    # Intersection des index (les periodes ne couvrent pas forcement la meme plage)
    hybrid = wind.join(solar, how="inner")
    hybrid = hybrid.join(df_prices, how="left")
    hybrid["price_eur_mwh"] = hybrid["price_eur_mwh"].interpolate().ffill().bfill()

    hybrid["P_hybrid_MW"] = hybrid["P_wind_MW"] + hybrid["P_solar_MW"]

    print(f"OK Hybride aligne : {len(hybrid):,} heures communes "
          f"({hybrid.index[0]} -> {hybrid.index[-1]})")
    return hybrid


# ===========================================================================
# 2. FEATURE ENGINEERING (calendaire + meteo)
# ===========================================================================

def make_features(df: pd.DataFrame, weather_cols=None) -> pd.DataFrame:
    """Construit les features pour le ML : cycles calendaires + meteo dispo."""
    X = pd.DataFrame(index=df.index)
    h = df.index.hour
    doy = df.index.dayofyear
    month = df.index.month
    dow = df.index.dayofweek

    # Encodage cyclique (evite la discontinuite 23h->0h)
    X["hour_sin"]  = np.sin(2 * np.pi * h / 24)
    X["hour_cos"]  = np.cos(2 * np.pi * h / 24)
    X["doy_sin"]   = np.sin(2 * np.pi * doy / 365)
    X["doy_cos"]   = np.cos(2 * np.pi * doy / 365)
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dow"]       = dow
    X["is_weekend"] = (dow >= 5).astype(int)

    # Features meteo si disponibles
    if weather_cols:
        for c in weather_cols:
            if c in df.columns:
                X[c] = df[c].values
    return X


# ===========================================================================
# 3. ENTRAINEMENT DES MODELES ML
# ===========================================================================

def train_model(df: pd.DataFrame, target: str, weather_cols=None,
                model_type="gbr", verbose=True):
    """
    Entraine un modele ML a predire `target` (ex: 'P_hybrid_MW' ou 'price_eur_mwh').
    Retourne (modele, metriques, features_utilisees).
    """
    X = make_features(df, weather_cols)
    y = df[target].values

    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=150, max_depth=18, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2" : r2_score(y_test, pred),
        "mean_target": float(np.mean(y)),
    }
    if verbose:
        print(f"  [{target}] {model_type.upper()} : "
              f"MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}  "
              f"(moy cible {metrics['mean_target']:.2f})")
    # On attache au modele l'ordre des colonnes et les moyennes d'entrainement,
    # pour reconstruire un jeu de features identique en prevision (meme sans meteo).
    model._feature_order = list(X.columns)
    model._feature_means = X.mean().to_dict()
    return model, metrics, list(X.columns)


# ===========================================================================
# 4. PREVISION FUTURE
# ===========================================================================

def forecast_future(model, future_index: pd.DatetimeIndex,
                    weather_future: pd.DataFrame = None,
                    weather_cols=None) -> pd.Series:
    """
    Predit la cible sur un index futur.

    Pour les variables meteo : si `weather_future` est fourni, on l'utilise ;
    sinon chaque colonne meteo manquante est remplie par sa MOYENNE d'entrainement
    (prevision "annee-type" climatologique). Le jeu de features final est
    REALIGNE exactement sur celui vu a l'entrainement (meme ordre, memes colonnes),
    ce qui evite l'erreur de feature names de scikit-learn.
    """
    base = pd.DataFrame(index=future_index)
    if weather_future is not None:
        base = base.join(weather_future, how="left")
    X = make_features(base, weather_cols)

    # Realignement strict sur les features d'entrainement
    feat_order = getattr(model, "_feature_order", list(X.columns))
    feat_means = getattr(model, "_feature_means", {})
    for col in feat_order:
        if col not in X.columns:
            # colonne meteo absente en prevision -> moyenne d'entrainement
            X[col] = feat_means.get(col, 0.0)
    # remplir les NaN residuels par la moyenne d'entrainement (ou 0)
    for col in feat_order:
        X[col] = X[col].fillna(feat_means.get(col, 0.0))
    X = X[feat_order]  # ordre identique a l'entrainement

    pred = model.predict(X)
    return pd.Series(pred, index=future_index)


def build_future_index(years_ahead=5, start_year=2025, freq="h"):
    """Cree un index horaire futur pour la projection."""
    start = f"{start_year}-01-01"
    end = f"{start_year + years_ahead - 1}-12-31 23:00"
    return pd.date_range(start, end, freq=freq)


# ===========================================================================
# 5. VALORISATION HYBRIDE
# ===========================================================================

def valorize_hybrid(df: pd.DataFrame) -> dict:
    """
    Valorise le parc hybride au prix horaire reel.
    Calcule revenus, prix de capture (wind/solar/hybrid), et profil journalier.
    df doit contenir : P_wind_MW, P_solar_MW, P_hybrid_MW, price_eur_mwh
    """
    out = {}
    d = df.copy()

    # Revenu horaire (EUR) = puissance (MW) * 1h * prix (EUR/MWh)
    d["rev_wind"]   = d["P_wind_MW"]   * d["price_eur_mwh"]
    d["rev_solar"]  = d["P_solar_MW"]  * d["price_eur_mwh"]
    d["rev_hybrid"] = d["P_hybrid_MW"] * d["price_eur_mwh"]

    n_years = (d.index[-1] - d.index[0]).days / 365.25
    out["n_years"] = n_years

    # Energie totale (MWh) et revenu total (M EUR)
    for tech in ["wind", "solar", "hybrid"]:
        e = d[f"P_{tech}_MW"].sum()                  # MWh (pas horaire = 1h)
        rev = d[f"rev_{tech}"].sum() / 1e6           # M EUR
        out[f"energy_{tech}_GWh_yr"] = e / 1000 / n_years
        out[f"revenue_{tech}_Meur_yr"] = rev / n_years
        # Prix de capture = revenu / energie (EUR/MWh)
        out[f"capture_price_{tech}"] = (rev * 1e6 / e) if e > 0 else 0.0

    # Prix marche moyen (baseload)
    out["baseload_price"] = float(d["price_eur_mwh"].mean())
    # Capture ratio (capture / baseload)
    for tech in ["wind", "solar", "hybrid"]:
        out[f"capture_ratio_{tech}"] = (
            out[f"capture_price_{tech}"] / out["baseload_price"] * 100
            if out["baseload_price"] else 0.0
        )

    # Profil journalier moyen (complementarite wind/solar)
    out["daily_profile"] = d.groupby(d.index.hour)[
        ["P_wind_MW", "P_solar_MW", "P_hybrid_MW"]
    ].mean()

    # Prix capte jour par jour (serie journaliere)
    daily = d.resample("D").agg(
        e_hybrid=("P_hybrid_MW", "sum"),
        rev_hybrid=("rev_hybrid", "sum"),
        price_mean=("price_eur_mwh", "mean"),
    )
    daily["capture_price_eur_mwh"] = np.where(
        daily["e_hybrid"] > 0, daily["rev_hybrid"] / daily["e_hybrid"], np.nan
    )
    out["daily"] = daily

    return out


def hybrid_lcoe(eco_wind: dict, eco_solar: dict,
                kpi_wind: dict, kpi_solar: dict,
                discount_rate=0.05, lifetime=30) -> dict:
    """
    LCOE du systeme hybride = (CAPEX + OPEX actualises des deux parcs)
                              / (production actualisee des deux parcs).
    """
    capex = eco_wind["capex_meur"] + eco_solar["capex_meur"]
    opex_yr = eco_wind["opex_meur_yr"] + eco_solar["opex_meur_yr"]

    r, n = discount_rate, lifetime
    annuity = (1 - (1 + r) ** (-n)) / r
    opex_disc = opex_yr * annuity

    # Production actualisee : somme des deux, chacune avec sa degradation
    prod_disc = eco_wind["prod_lifetime_gwh"] + eco_solar["prod_lifetime_gwh"]
    # (prod_lifetime_gwh = somme non actualisee ; pour LCOE on actualise via les
    #  routines de chaque module. On reprend ici une approche homogene simple :)
    # Recalage : on actualise la production P50 annuelle de chaque techno.
    def disc_prod(p50, deg_pct):
        yrs = np.arange(1, n + 1)
        deg = (1 - deg_pct / 100) ** yrs
        return sum(p50 * d / (1 + r) ** y for y, d in zip(yrs, deg))

    prod_disc = (
        disc_prod(kpi_wind["P50_GWh_yr"], 0.30)
        + disc_prod(kpi_solar["P50_GWh_yr"], 0.35)
    )

    lcoe = (capex + opex_disc) / prod_disc * 1000  # EUR/MWh
    return {
        "capex_total_meur": capex,
        "opex_total_meur_yr": opex_yr,
        "prod_lifetime_disc_gwh": prod_disc,
        "lcoe_hybrid_eur_mwh": lcoe,
    }
