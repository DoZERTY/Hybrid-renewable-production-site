from fetch_solar_data import fetch_weather_solar, fetch_electricity_prices, SITE, PERIOD, PARK, ECONOMICS
from compute_solar_production import compute_pv_production, analyze_solar_resource, compute_economics
from visualisation_solar import plot_solar_resource

# Récupérer les données météo et d'irradiation solaire
solar_df = fetch_weather_solar()

# Récupérer les données de prix de l'électricité
prices_df = fetch_electricity_prices()

# Fusionner les données sur la base de l'index temporel
combined_df = solar_df.join(prices_df, how="inner")

# Calculer la production solaire estimée
df_pv = compute_pv_production(combined_df)

# Analyser la ressource solaire et son potentiel de production
kpi_solar = analyze_solar_resource(df_pv)

# Analyser les aspects économiques de la production solaire par rapport aux prix de l'électricité
eco = compute_economics(df_pv, prices_df, kpi_solar)

plot_solar_resource(df_pv, kpi_solar, eco)