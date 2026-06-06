import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_simulation_30y(sims: dict, valorisations: dict):
    """
    sims          : {scenario: df_sim}  (sortie de simulate_30y)
    valorisations : {scenario: results} (sortie de valorize_contracts)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Simulation 30 ans - Scenarios RTE x Modes de valorisation",
                 fontsize=13, fontweight="bold")

    colors = {"bas": "tab:red", "central": "tab:orange", "haut": "tab:green"}

    # --- 1. Trajectoire du prix moyen annuel par scenario ---
    ax = axes[0, 0]
    for scen, df in sims.items():
        annual_price = df.groupby(df.index.year)["price_eur_mwh"].mean()
        ax.plot(annual_price.index, annual_price.values,
                color=colors.get(scen, "gray"), marker=".", label=f"Scenario {scen}")
    ax.set_title("Prix moyen annuel projete (scenarios RTE)")
    ax.set_xlabel("Annee")
    ax.set_ylabel("Prix (EUR/MWh)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # --- 2. Production annuelle (degradation) ---
    ax = axes[0, 1]
    df0 = list(sims.values())[0]
    annual_prod = df0.groupby(df0.index.year)["P_MW"].sum() / 1000
    ax.bar(annual_prod.index, annual_prod.values, color="tab:blue", alpha=0.7)
    ax.set_title("Production annuelle nette (avec degradation)")
    ax.set_xlabel("Annee")
    ax.set_ylabel("Production (GWh/an)")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # --- 3. NPV par mode et par scenario (barres groupees) ---
    ax = axes[1, 0]
    scenarios = list(valorisations.keys())
    modes = ["spot", "cfd", "ppa"]
    mode_labels = ["Spot", "CfD", "PPA"]
    x = np.arange(len(scenarios))
    width = 0.25
    for i, (mode, lab) in enumerate(zip(modes, mode_labels)):
        npvs = [valorisations[s][mode]["npv_meur"] for s in scenarios]
        ax.bar(x + (i - 1) * width, npvs, width, label=lab)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("NPV par mode de valorisation et scenario")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel("NPV (M EUR)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # --- 4. Prix effectif capte par mode (scenario central) ---
    ax = axes[1, 1]
    central = valorisations.get("central", list(valorisations.values())[0])
    prices_eff = [central[m]["prix_effectif_eur_mwh"] for m in modes]
    bars = ax.bar(mode_labels, prices_eff,
                  color=["tab:gray", "tab:blue", "tab:green"])
    for b, v in zip(bars, prices_eff):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Prix effectif capte par mode (scenario central)")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.show()
