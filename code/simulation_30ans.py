"""
================================================================================
SIMULATION 30 ANS - Production ML + Scenarios prix RTE + Valorisation CfD/PPA
================================================================================

Brique 1 : ML apprend la production horaire (forme deterministe depuis la meteo)
           et la rejoue sur 30 ans (annee-type repetee, avec degradation).
Brique 2 : Prix - on PLAFONNE les extremes (winsorisation P1/P99) pour neutraliser
           la crise 2022, puis le ML apprend la FORME horaire/saisonniere du prix.
Brique 3 : Le NIVEAU du prix futur vient des SCENARIOS RTE (bas/central/haut),
           pas du ML (qui ne sait pas extrapoler une tendance macro 30 ans).
           On ajoute une montee des heures negatives (penetration EnR croissante).
Brique 4 : Valorisation comparee CfD vs PPA vs spot.

HYPOTHESES DE PRIX (a sourcer precisement pour le dossier final) :
  Ordres de grandeur RTE Bilan previsionnel 2023-2035 : prix de marche projetes
  ~35-50 EUR/MWh (decarbonation), ~70 EUR/MWh en post-crise gaziere.
  -> scenarios retenus ci-dessous, PARAMETRABLES.

NB : ce module est independant. Il prend en entree les DataFrames deja produits
par la chaine principale (df_wind, df_solar, prices_df reels ENTSO-E).
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ===========================================================================
# SCENARIOS DE PRIX RTE (niveau moyen EUR/MWh par horizon) - PARAMETRABLES
# ===========================================================================
# Prix de marche moyen baseload projete. Interpolation lineaire entre jalons.
SCENARIOS_RTE = {
    "bas": {2025: 60, 2030: 45, 2040: 40, 2050: 38},      # decarbonation rapide, surcapacite
    "central": {2025: 70, 2030: 58, 2040: 55, 2050: 52},  # trajectoire mediane
    "haut": {2025: 85, 2030: 75, 2040: 72, 2050: 70},     # tensions gaz/demande forte
}

# Montee des heures de prix negatif (part de la production exposee), par scenario.
# Cale sur la tendance observee 2023-2024 (forte hausse penetration EnR).
NEG_HOURS_GROWTH = {
    "bas": 0.12,      # +12%/an d'heures negatives (penetration EnR forte)
    "central": 0.08,
    "haut": 0.04,     # moins de negatifs si demande forte
}


def niveau_prix_annee(scenario: str, annee: int) -> float:
    """Interpole le niveau de prix moyen RTE pour une annee donnee."""
    jalons = SCENARIOS_RTE[scenario]
    annees = sorted(jalons.keys())
    if annee <= annees[0]:
        return jalons[annees[0]]
    if annee >= annees[-1]:
        return jalons[annees[-1]]
    return float(np.interp(annee, annees, [jalons[a] for a in annees]))


# ===========================================================================
# BRIQUE 1 + 2 : ENTRAINEMENT ML (production + FORME du prix)
# ===========================================================================

def _make_features(index: pd.DatetimeIndex, extra: pd.DataFrame = None) -> pd.DataFrame:
    """Features calendaires cycliques (+ meteo optionnelle)."""
    X = pd.DataFrame(index=index)
    h, doy, m, dow = index.hour, index.dayofyear, index.month, index.dayofweek
    X["hour_sin"] = np.sin(2*np.pi*h/24);   X["hour_cos"] = np.cos(2*np.pi*h/24)
    X["doy_sin"]  = np.sin(2*np.pi*doy/365); X["doy_cos"] = np.cos(2*np.pi*doy/365)
    X["month_sin"]= np.sin(2*np.pi*m/12);   X["month_cos"]= np.cos(2*np.pi*m/12)
    X["is_weekend"] = (dow >= 5).astype(int)
    if extra is not None:
        for c in extra.columns:
            X[c] = extra[c].values
    return X


def train_production_model(df_prod: pd.DataFrame, power_col="P_AC_MW",
                           weather_cols=None) -> dict:
    """ML qui apprend la production horaire. Retourne modele + features."""
    extra = df_prod[weather_cols] if weather_cols else None
    X = _make_features(df_prod.index, extra)
    # Securise les features meteo (NaN eventuels) avant entrainement
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean()).fillna(0.0)
    y = df_prod[power_col].values
    mask = ~np.isnan(y)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=42)
    model.fit(X[mask], y[mask])
    pred = model.predict(X[mask])
    print(f"  [Production] R2={r2_score(y[mask], pred):.3f} "
          f"MAE={mean_absolute_error(y[mask], pred):.1f} MW")
    return {"model": model, "feat_order": list(X.columns),
            "feat_means": X.mean().to_dict(), "weather_cols": weather_cols or []}


def train_price_shape_model(df_prices: pd.DataFrame, price_col="price_eur_mwh",
                            cap_low=0.01, cap_high=0.99) -> dict:
    """
    ML qui apprend la FORME horaire/saisonniere du prix (normalisee a moyenne 1).
    Les prix extremes sont plafonnes (winsorisation) AVANT apprentissage.
    """
    p = df_prices[price_col].copy()
    # Nettoyage prealable : ENTSO-E peut avoir des heures manquantes (NaN) sur 10 ans
    n_nan_in = p.isna().sum()
    if n_nan_in > 0:
        print(f"  [Prix] {n_nan_in} NaN dans les prix bruts -> interpolation")
        p = p.interpolate().ffill().bfill()

    lo, hi = p.quantile([cap_low, cap_high])
    p_capped = p.clip(lo, hi)
    print(f"  [Prix] plafonnement P{cap_low*100:.0f}={lo:.0f} / "
          f"P{cap_high*100:.0f}={hi:.0f} EUR/MWh "
          f"(moy {p.mean():.1f}->{p_capped.mean():.1f}, max {p.max():.0f}->{p_capped.max():.0f})")

    # Forme = prix / moyenne annuelle (enleve le niveau, garde le motif)
    annual_mean = p_capped.groupby(p_capped.index.year).transform("mean")
    # Garde-fou contre division par ~0 : on remplace les moyennes nulles par 1
    annual_mean = annual_mean.replace(0, np.nan).fillna(p_capped.mean() or 1.0)
    shape = (p_capped / annual_mean)
    # Nettoyage final : infinis -> NaN -> interpolation, puis clip
    shape = shape.replace([np.inf, -np.inf], np.nan).interpolate().ffill().bfill()
    shape = shape.fillna(1.0).clip(-2, 5)

    X = _make_features(shape.index)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=42)
    model.fit(X, shape.values)
    pred = model.predict(X)
    print(f"  [Prix-forme] R2={r2_score(shape.values, pred):.3f}")
    return {"model": model, "feat_order": list(X.columns),
            "feat_means": X.mean().to_dict()}


def _predict(model_dict: dict, index: pd.DatetimeIndex,
             extra: pd.DataFrame = None) -> pd.Series:
    """Predit en realignant strictement les features sur l'entrainement."""
    X = _make_features(index, extra)
    for col in model_dict["feat_order"]:
        if col not in X.columns:
            X[col] = model_dict["feat_means"].get(col, 0.0)
        X[col] = X[col].fillna(model_dict["feat_means"].get(col, 0.0))
    X = X[model_dict["feat_order"]]
    return pd.Series(model_dict["model"].predict(X), index=index)


# ===========================================================================
# BRIQUE 3 : SIMULATION 30 ANS (production degradee + prix scenario RTE)
# ===========================================================================

def simulate_30y(prod_model: dict, price_shape_model: dict,
                 scenario: str = "central",
                 start_year: int = 2025, n_years: int = 30,
                 degradation_pct_yr: float = 0.30,
                 weather_typical: pd.DataFrame = None,
                 df_prod_hist: pd.DataFrame = None,
                 power_col: str = "P_AC_MW") -> pd.DataFrame:
    """
    Simule production + prix horaires sur n_years.

    PRODUCTION - deux methodes :
      (A) BOOTSTRAP HISTORIQUE (recommande, defaut si df_prod_hist fourni) :
          on tire au hasard des annees historiques REELLES et on les rejoue.
          -> preserve le vrai facteur de charge et la variabilite interannuelle.
      (B) ANNEE-TYPE ML (si weather_typical fourni, sinon) :
          le ML rejoue un profil lisse. ATTENTION : lisser la meteo avant la
          courbe de puissance non-lineaire SURESTIME le facteur de charge.
          A utiliser seulement faute de mieux.

    PRIX : forme ML (horaire/saisonniere) x niveau RTE de l'annee + heures negatives.
    """
    future_idx = pd.date_range(f"{start_year}-01-01",
                               f"{start_year + n_years - 1}-12-31 23:00", freq="h")

    if df_prod_hist is not None:
        # ---- Methode A : bootstrap des annees historiques reelles ----
        hist_years = sorted(set(df_prod_hist.index.year))
        rng = np.random.default_rng(42)
        prod_values = np.empty(len(future_idx))
        for y in range(start_year, start_year + n_years):
            src_year = rng.choice(hist_years)
            src = df_prod_hist[df_prod_hist.index.year == src_year][power_col].values
            mask_y = future_idx.year == y
            n_h = int(mask_y.sum())
            prod_values[mask_y] = np.resize(src, n_h)  # ajuste longueur (bissextiles)
        prod = pd.Series(prod_values, index=future_idx).clip(lower=0)
    else:
        # ---- Methode B : annee-type ML ----
        prod = _predict(prod_model, future_idx, weather_typical).clip(lower=0)

    # Degradation : multiplie par (1 - deg)^(annee - start)
    years_elapsed = future_idx.year - start_year
    deg_factor = (1 - degradation_pct_yr / 100) ** years_elapsed
    prod = prod * deg_factor

    # Forme du prix (motif normalise ~1 en moyenne)
    shape = _predict(price_shape_model, future_idx)

    # Niveau RTE par annee
    niveau = np.array([niveau_prix_annee(scenario, y) for y in future_idx.year])
    price = shape.values * niveau

    # Montee des heures negatives : on force une fraction croissante d'heures
    # (les plus basses) a devenir negatives, proportionnelle a la penetration EnR.
    growth = NEG_HOURS_GROWTH[scenario]
    df = pd.DataFrame({"P_MW": prod.values, "price_eur_mwh": price}, index=future_idx)
    for y in range(start_year, start_year + n_years):
        mask_y = df.index.year == y
        # part d'heures negatives cible cette annee (base 4% en 2025, croissance composee)
        base_neg = 0.04
        target_neg = min(base_neg * (1 + growth) ** (y - start_year), 0.25)
        sub = df.loc[mask_y, "price_eur_mwh"]
        n_neg = int(len(sub) * target_neg)
        if n_neg > 0:
            seuil = sub.nsmallest(n_neg).max()
            # les heures sous le seuil basculent vers du negatif (proportionnel)
            neg_mask = mask_y & (df["price_eur_mwh"] <= seuil)
            df.loc[neg_mask, "price_eur_mwh"] = -np.abs(
                df.loc[neg_mask, "price_eur_mwh"] * 0.3
            )

    return df


# ===========================================================================
# BRIQUE 4 : VALORISATION CfD vs PPA vs SPOT
# ===========================================================================

def valorize_contracts(df_sim: pd.DataFrame,
                        cfd_strike: float = 90.0,
                        ppa_price: float = 80.0,
                        discount_rate: float = 0.0474,
                        capex_meur: float = 854.0,
                        opex_meur_yr: float = 15.3,
                        power_col: str = "P_MW") -> dict:
    """
    Compare 3 modes de valorisation sur la simulation 30 ans :
      - SPOT  : vente au prix de marche horaire (expose a la volatilite et aux negatifs)
      - CfD   : prix garanti 'cfd_strike'. L'Etat complete si spot < strike,
                le producteur reverse si spot > strike. Revenu = strike x energie
                (sauf heures negatives ou souvent la production est arretee/non remuneree).
      - PPA   : prix fixe 'ppa_price' sur toute l'energie (contrat prive long terme).
    Retourne revenus actualises, NPV, et prix moyen effectif par mode.
    """
    d = df_sim.copy()
    start_year = d.index.year.min()
    r = discount_rate

    results = {"scenario_params": {
        "cfd_strike": cfd_strike, "ppa_price": ppa_price,
        "discount_rate": r, "capex_meur": capex_meur, "opex_meur_yr": opex_meur_yr}}

    # Revenu horaire par mode
    d["rev_spot"] = d[power_col] * d["price_eur_mwh"]
    # CfD : prix garanti, mais on ne remunere pas l'energie produite a prix negatif
    # (regle frequente des CfD recents : pas de complement si prix spot < 0)
    cfd_price = np.where(d["price_eur_mwh"] < 0, 0.0, cfd_strike)
    d["rev_cfd"] = d[power_col] * cfd_price
    # PPA : prix fixe sur toute l'energie
    d["rev_ppa"] = d[power_col] * ppa_price

    # Agregation annuelle + actualisation
    for mode in ["spot", "cfd", "ppa"]:
        annual = d.groupby(d.index.year)[f"rev_{mode}"].sum() / 1e6  # M EUR/an
        years = annual.index.values
        disc = annual.values / (1 + r) ** (years - start_year + 1)
        opex_disc = sum(opex_meur_yr / (1 + r) ** (y - start_year + 1) for y in years)
        npv = disc.sum() - opex_disc - capex_meur
        energy_tot = d[power_col].sum() / 1000  # GWh sur la periode
        rev_tot = d[f"rev_{mode}"].sum() / 1e6
        results[mode] = {
            "revenue_total_meur": rev_tot,
            "revenue_mean_meur_yr": float(annual.mean()),
            "npv_meur": npv,
            "prix_effectif_eur_mwh": rev_tot * 1e6 / (energy_tot * 1000) if energy_tot else 0,
        }

    return results


def print_comparison(results: dict):
    """Affiche le tableau comparatif des 3 modes."""
    p = results["scenario_params"]
    print("\n" + "=" * 68)
    print("  COMPARAISON VALORISATION 30 ANS")
    print(f"  CfD strike={p['cfd_strike']:.0f} | PPA={p['ppa_price']:.0f} | "
          f"WACC={p['discount_rate']*100:.2f}% | CAPEX={p['capex_meur']:.0f} M EUR")
    print("=" * 68)
    print(f"  {'Mode':<8} {'Revenu/an':>12} {'Prix effectif':>15} {'NPV':>12}")
    print(f"  {'':8} {'(M EUR)':>12} {'(EUR/MWh)':>15} {'(M EUR)':>12}")
    print("  " + "-" * 64)
    for mode, label in [("spot", "SPOT"), ("cfd", "CfD"), ("ppa", "PPA")]:
        m = results[mode]
        print(f"  {label:<8} {m['revenue_mean_meur_yr']:>12.1f} "
              f"{m['prix_effectif_eur_mwh']:>15.1f} {m['npv_meur']:>12.0f}")
    print("=" * 68)
