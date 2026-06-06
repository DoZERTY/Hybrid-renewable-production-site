import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hybrid(hybrid: pd.DataFrame, val: dict, lcoe: dict,
                forecast: pd.DataFrame = None):
    """
    Figures du systeme hybride :
      - Complementarite journaliere wind/solar
      - Prix de capture par techno vs baseload
      - Mix de production annuel
      - Prix capte jour par jour (serie)
      - Prevision future (si fournie)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Systeme Hybride Eolien + Solaire - Valorisation au prix reel\n"
        f"LCOE hybride : {lcoe['lcoe_hybrid_eur_mwh']:.1f} EUR/MWh | "
        f"Capture hybride : {val['capture_price_hybrid']:.1f} EUR/MWh "
        f"({val['capture_ratio_hybrid']:.0f}% du baseload)",
        fontsize=12
    )

    # --- 1. Complementarite journaliere ---
    ax = axes[0, 0]
    dp = val["daily_profile"]
    ax.plot(dp.index, dp["P_wind_MW"], label="Eolien", color="tab:blue", marker=".")
    ax.plot(dp.index, dp["P_solar_MW"], label="Solaire", color="tab:orange", marker=".")
    ax.plot(dp.index, dp["P_hybrid_MW"], label="Hybride", color="tab:green",
            linewidth=2.2)
    ax.fill_between(dp.index, dp["P_hybrid_MW"], alpha=0.12, color="tab:green")
    ax.set_title("Complementarite Journaliere Moyenne")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Puissance (MW)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 2. Prix de capture par techno ---
    ax = axes[0, 1]
    techs = ["wind", "solar", "hybrid"]
    labels = ["Eolien", "Solaire", "Hybride"]
    captures = [val[f"capture_price_{t}"] for t in techs]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    bars = ax.bar(labels, captures, color=colors)
    ax.axhline(val["baseload_price"], color="tab:red", linestyle="--",
               label=f"Baseload: {val['baseload_price']:.1f}")
    for b, c in zip(bars, captures):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.5, f"{c:.0f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Prix de Capture par Technologie")
    ax.set_ylabel("EUR/MWh")
    ax.legend(fontsize=8)

    # --- 3. Mix de production annuel ---
    ax = axes[0, 2]
    energies = [val["energy_wind_GWh_yr"], val["energy_solar_GWh_yr"]]
    ax.pie(energies, labels=["Eolien", "Solaire"],
           colors=["tab:blue", "tab:orange"], autopct="%1.0f%%",
           startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title(f"Mix Energetique Hybride\n"
                 f"({val['energy_hybrid_GWh_yr']:.0f} GWh/an total)")

    # --- 4. Prix capte jour par jour ---
    ax = axes[1, 0]
    daily = val["daily"]
    ax.plot(daily.index, daily["capture_price_eur_mwh"],
            color="tab:green", alpha=0.5, linewidth=0.6, label="Capture jour")
    # moyenne glissante 30j
    roll = daily["capture_price_eur_mwh"].rolling(30, min_periods=7).mean()
    ax.plot(daily.index, roll, color="darkgreen", linewidth=1.8,
            label="Moyenne glissante 30j")
    ax.plot(daily.index, daily["price_mean"].rolling(30, min_periods=7).mean(),
            color="tab:red", linewidth=1.2, linestyle="--",
            label="Prix marche 30j")
    ax.set_title("Prix Capte Hybride - Jour par Jour")
    ax.set_xlabel("Date")
    ax.set_ylabel("EUR/MWh")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 5. Capture ratio (cannibalisation) ---
    ax = axes[1, 1]
    ratios = [val[f"capture_ratio_{t}"] for t in techs]
    bars = ax.bar(labels, ratios, color=colors)
    ax.axhline(100, color="tab:red", linestyle="--", label="Baseload (100%)")
    for b, rr in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, rr + 1, f"{rr:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Ratio de Capture (vs Baseload)")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)

    # --- 6. Prevision future ---
    ax = axes[1, 2]
    if forecast is not None and len(forecast) > 0:
        fc_monthly = forecast.resample("ME").agg(
            prod=("P_hybrid_MW", "sum"), price=("price_eur_mwh", "mean")
        )
        fc_monthly["prod"] /= 1000  # GWh
        ax.bar(fc_monthly.index, fc_monthly["prod"], width=20,
               color="tab:green", alpha=0.6, label="Prod prevue (GWh)")
        ax2 = ax.twinx()
        ax2.plot(fc_monthly.index, fc_monthly["price"], color="tab:red",
                 marker=".", label="Prix prevu (EUR/MWh)")
        ax.set_title("Prevision ML - Mensuelle")
        ax.set_xlabel("Date")
        ax.set_ylabel("Production (GWh)")
        ax2.set_ylabel("Prix (EUR/MWh)")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Prevision non disponible", ha="center", va="center")

    plt.tight_layout()
    plt.show()
