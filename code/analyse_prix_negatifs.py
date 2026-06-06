"""
================================================================================
ANALYSE DES PRIX NEGATIFS - Parc Leucate
================================================================================
A inserer dans main.py, ou a lancer separement.

Repond a la question : "pourquoi les prix negatifs n'apparaissent pas sur les
graphes ?" -> ils sont noyes par l'echelle (pic 2022 a 700 EUR) et lisses par
les moyennes journalieres/mensuelles. Ce module les isole et les quantifie.

Usage dans main.py, apres avoir recupere df_wind et prices_df :
    from analyse_prix_negatifs import analyze_negative_prices, plot_negative_prices
    analyze_negative_prices(df_wind, prices_df)
    plot_negative_prices(prices_df)   # graphe dedie, echelle adaptee
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_negative_prices(df_prod: pd.DataFrame,
                            df_prices: pd.DataFrame,
                            power_col: str = "P_AC_MW") -> pd.DataFrame:
    """
    Quantifie les prix negatifs et leur impact sur la production du parc.
    Affiche un rapport texte. Retourne le DataFrame croise prod+prix.
    """
    d = df_prod[[power_col]].join(df_prices, how="left")
    d["price_eur_mwh"] = d["price_eur_mwh"].interpolate()
    d["revenue"] = d[power_col] * d["price_eur_mwh"]

    neg = d[d["price_eur_mwh"] < 0]
    n_neg, n_tot = len(neg), len(d)
    n_years = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)

    print("=" * 60)
    print("  ANALYSE DES PRIX NEGATIFS (donnees ENTSO-E reelles)")
    print("=" * 60)
    print(f"  Heures de prix negatif : {n_neg:,} / {n_tot:,} "
          f"({n_neg / n_tot * 100:.2f}%) soit {n_neg / n_years:.0f} h/an")
    print(f"  Prix minimum observe   : {d['price_eur_mwh'].min():.1f} EUR/MWh")
    print(f"  Prix maximum observe   : {d['price_eur_mwh'].max():.1f} EUR/MWh")

    if n_neg > 0:
        e_neg = neg[power_col].sum()
        e_tot = d[power_col].sum()
        perte = neg["revenue"].sum() / 1e6
        print(f"  Energie vendue a prix<0: {e_neg / 1000:.0f} GWh "
              f"({e_neg / e_tot * 100:.2f}% de la prod totale)")
        print(f"  Manque a gagner net    : {perte:.2f} M EUR sur "
              f"{n_years:.0f} ans ({perte / n_years:.2f} M EUR/an)")
        print(f"  Prod moyenne qd prix<0 : {neg[power_col].mean():.0f} MW "
              f"(vs {d[power_col].mean():.0f} MW en moyenne)")
        # Repartition par annee
        neg_by_year = neg.groupby(neg.index.year).size()
        print("\n  Heures negatives par annee :")
        for yr, cnt in neg_by_year.items():
            print(f"    {yr} : {cnt:>4} h")
    else:
        print("  Aucune heure de prix negatif dans les donnees.")

    return d


def plot_negative_prices(df_prices: pd.DataFrame):
    """
    Graphe dedie qui REND VISIBLES les prix negatifs (echelle adaptee,
    contrairement au graphe global ecrase par le pic 2022).
    """
    p = df_prices["price_eur_mwh"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Visibilite des Prix Negatifs (ENTSO-E)", fontsize=12)

    # --- Gauche : histogramme zoom sur la zone basse ---
    ax = axes[0]
    ax.hist(p[p < 50], bins=80, color="tab:blue", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", label="Prix = 0")
    ax.set_title("Distribution des prix bas (< 50 EUR/MWh)")
    ax.set_xlabel("Prix (EUR/MWh)")
    ax.set_ylabel("Nombre d'heures")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- Droite : nombre d'heures negatives par an ---
    ax = axes[1]
    neg = p[p < 0]
    if len(neg) > 0:
        by_year = neg.groupby(neg.index.year).size()
        ax.bar(by_year.index, by_year.values, color="tab:red", alpha=0.7)
        ax.set_title("Heures de prix negatif par annee")
        ax.set_xlabel("Annee")
        ax.set_ylabel("Heures")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Aucun prix negatif", ha="center", va="center")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test autonome avec prix synthetiques incluant des negatifs
    from fetch_wind_data import _generate_synthetic_prices
    prices = _generate_synthetic_prices("2015-01-01", "2024-12-31")
    idx = prices.index
    prod = pd.DataFrame({"P_AC_MW": np.random.uniform(0, 213, len(idx))}, index=idx)
    analyze_negative_prices(prod, prices)
    plot_negative_prices(prices)
