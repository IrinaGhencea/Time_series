import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

file_path = "Dataset/temperature.csv"
df = pd.read_csv(file_path)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)
temperature = df.groupby(df.index).mean(numeric_only=True)



# Average temperature in time
plt.figure(figsize=(12, 6))
plt.plot(temperature.index, temperature['Avg_Temperature_degC'], label="Avg Temperature (celsius)", color='blue')
plt.xlabel("Year")
plt.ylabel("Temperature(celsius)")
plt.title("Global Average Temperature Trends")
plt.legend()
plt.grid()
plt.savefig("Dataset/global_avg_temperature_trends.png")
plt.close()


# Moving average
temperature['Temp_Moving_Avg'] =temperature['Avg_Temperature_degC'].rolling(window=5).mean()
plt.figure(figsize=(12, 6))
plt.plot(temperature.index, temperature['Avg_Temperature_degC'], label="Original Data", alpha=0.5)
plt.plot(temperature.index, temperature['Temp_Moving_Avg'], label="5 Year moving average", color='red')
plt.xlabel("Year")
plt.ylabel("Temperature (celsius)")
plt.title("Temperature with moving average")
plt.legend()
plt.grid()
plt.savefig("Dataset/temperature_with_moving_average.png")
plt.close()


# CO2 emissions and sea level rise
plt.figure(figsize=(12, 6))
plt.plot(temperature.index, temperature['CO2_Emissions_tons_per_capita'], label="CO₂ Emissions", color='green')
plt.plot(temperature.index, temperature['Sea_Level_Rise_mm'], label="Sea Level Rise", color='purple')
plt.xlabel("Year")
plt.ylabel("Value")
plt.title("CO₂ emissions and Sea level rise")
plt.legend()
plt.grid()
plt.savefig("Dataset/co2_and_sea_level_rise.png")
plt.close()


# Augmented Dickey-Fuller Test on temperature
adf_test = adfuller(temperature['Avg_Temperature_degC'].dropna())
print("\nAugmented Dickey-Fuller Test Results for temperature ")
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
if adf_test[1]<0.05:
    print("The time series is stationary")
else:
    print("The time series is not stationary")


# Correlation analysis
correlation = temperature[['Avg_Temperature_degC', 'CO2_Emissions_tons_per_capita', 'Sea_Level_Rise_mm']].corr()
print("\nCorrelation between temperature, CO₂ emissions and sea level rise")
print(correlation)
