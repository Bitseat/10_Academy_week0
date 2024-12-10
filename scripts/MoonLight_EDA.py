import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Summary Statistics: Calculate the mean, median, standard deviation, and other statistical measures for each numeric column to understand data distribution.

def process_file(file_path):

    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=np.number).columns[:-1]
    summary_stats = df[numeric_cols].describe().T
    return summary_stats

# List of file paths
file_paths = ["data/benin-malanville.csv"] #, "data/sierraleone-bumbuna.csv", "data/togo-dapaong_qc.csv"]

# Process each file and store the results
summary_stats_list = []
for file_path in file_paths:
    summary_stats = process_file(file_path)
    summary_stats_list.append(summary_stats)

# Combine the results into a single DataFrame
combined_summary_stats = pd.concat(summary_stats_list, keys=file_paths)

# Print the combined summary statistics
print(combined_summary_stats)

# Data Quality Check: Look for missing values, outliers, or incorrect entries (e.g., negative values where only positive should exist), especially in columns like GHI, DNI, and DHI and check for outliers, especially in sensor readings (ModA, ModB) and wind speed data (WS, WSgust).
def check_data_quality(file_path):

    df = pd.read_csv(file_path)

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values in {file_path}:\n{missing_values}")

    # Check for negative values in GHI, DNI, and DHI
    negative_values = df[(df['GHI'] < 0) | (df['DNI'] < 0) | (df['DHI'] < 0)]
    if not negative_values.empty:
        print(f"Negative values found in GHI, DNI, or DHI in {file_path}:\n{negative_values}")

    # Check for outliers in sensor readings and wind speed data
    numeric_cols = ['ModA', 'ModB', 'WS', 'WSgust']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"Outliers found in {col} in {file_path}:\n{outliers}")

for file_path in file_paths:
    check_data_quality(file_path)

def clean_data(file_path):
    df = pd.read_csv(file_path)

    # # Remove rows with missing values
    # df.dropna(inplace=True)

    # Remove rows with negative values in GHI, DNI, and DHI
    df = df[(df['GHI'] >= 0) & (df['DNI'] >= 0) & (df['DHI'] >= 0)]

    # Remove outliers in sensor readings and wind speed data
    numeric_cols = ['ModA', 'ModB', 'WS', 'WSgust']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df

for file_path in file_paths:
    cleaned_df = clean_data(file_path)
    cleaned_df.to_csv(f"cleaned_benin.csv", index=False)

# Time Series Analysis: Plot bar charts or line charts  of GHI, DNI, DHI, and Tamb over time to observe patterns by month, trends throughout day, or anomalies, such as peaks in solar irradiance or temperature fluctuations. 
# Read the CSV data into a DataFrame
cleaned_df = pd.read_csv('cleaned_benin.csv', parse_dates=['Timestamp'])

# Convert the 'Timestamp' column to a datetime index
cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
cleaned_df.set_index('Timestamp', inplace=True)

# Plot daily GHI, DNI, DHI, and Tamb
plt.figure(figsize=(12, 8))
plt.plot(cleaned_df['GHI'], label='GHI')
plt.plot(cleaned_df['DNI'], label='DNI')
plt.plot(cleaned_df['DHI'], label='DHI')
plt.plot(cleaned_df['Tamb'], label='Tamb')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Daily Variation of Solar Irradiance and Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Plot monthly average GHI, DNI, DHI, and Tamb
monthly_data = cleaned_df.resample('M').mean()
plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, monthly_data['GHI'], label='GHI')
plt.plot(monthly_data.index, monthly_data['DNI'], label='DNI')
plt.plot(monthly_data.index, monthly_data['DHI'], label='DHI')
plt.plot(monthly_data.index, monthly_data['Tamb'], label='Tamb')
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.title('Monthly Average Solar Irradiance and Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Plot hourly average GHI, DNI, DHI, and Tamb
hourly_data = cleaned_df.resample('H').mean()
hourly_avg_data = hourly_data.groupby(hourly_data.index.hour).mean()
plt.figure(figsize=(12, 6))
plt.plot(hourly_avg_data.index, hourly_avg_data['GHI'], label='GHI')
plt.plot(hourly_avg_data.index, hourly_avg_data['DNI'], label='DNI')
plt.plot(hourly_avg_data.index, hourly_avg_data['DHI'], label='DHI')
plt.plot(hourly_avg_data.index, hourly_avg_data['Tamb'], label='Tamb')
plt.xlabel('Hour')
plt.ylabel('Average Value')
plt.title('Hourly Average Solar Irradiance and Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Correlation Analysis: Use correlation matrices or pair plots to visualize the correlations between solar radiation components (GHI, DNI, DHI) and temperature measures (TModA, TModB). Investigate the relationship between wind conditions (WS, WSgust, WD) and solar irradiance using scatter matrices.
corr_matrix = cleaned_df[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Solar Radiation and Temperature')
plt.show()

# Pair Plot for Solar Radiation and Temperature
sns.pairplot(cleaned_df[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']])
plt.suptitle('Pair Plot for Solar Radiation and Temperature')
plt.show()

# Scatter Matrix for Wind Conditions and Solar Irradiance
sns.pairplot(cleaned_df[['WS', 'WSgust', 'WD', 'GHI', 'DNI', 'DHI']])
plt.suptitle('Scatter Matrix for Wind Conditions and Solar Irradiance')
plt.show()

# def plot_wind_rose(df, ws_col='WS', wd_col='WD'):
#     """
#     Plots a wind rose to visualize wind speed and direction.

#     Args:
#         df (pandas.DataFrame): The DataFrame containing wind speed and direction data.
#         ws_col (str, optional): The column name for wind speed. Defaults to 'WS'.
#         wd_col (str, optional): The column name for wind direction. Defaults to 'WD'.
#     """

#     # Create a polar plot
#     plt.figure(figsize=(8, 8))
#     ax = plt.subplot(111, polar=True)

#     # Define wind speed bins
#     ws_bins = [0, 2, 4, 6, 8, 10, 12]

#     # Calculate wind direction and speed frequencies
#     wind_data = df[[ws_col, wd_col]]
#     wind_data['wd_bin'] = pd.cut(wind_data[wd_col], bins=range(0, 361, 10))

#     wind_data['wd_bin'] = pd.cut(wind_data[wd_col], bins=range(0, 361, 10), labels=range(0, 36 // 10))  # Assign labels for clarity

#     wind_freq = wind_data.groupby('wd_bin').size()
#     wind_speed_freq = wind_data.groupby(['wd_bin', pd.cut(wind_data[ws_col], bins=ws_bins)]).size()

#     # Normalize wind speed frequencies
#     wind_speed_freq = wind_speed_freq.groupby(level=0).apply(lambda x: x / x.sum())

#     # Plot wind speed frequencies for each wind direction bin
#     colors = cm.viridis(np.linspace(0, 1, len(ws_bins) - 1))
#     for i in range(len(ws_bins) - 1):
#         ax.bar(wind_freq.index, wind_speed_freq.loc[:, (ws_bins[i], ws_bins[i + 1])],
#                width=10, bottom=sum(wind_speed_freq.loc[:, (ws_bins[j], ws_bins[j + 1])] for j in range(i)),
#                color=colors[i], edgecolor='k')

#     # Set plot labels and title
#     ax.set_theta_zero_location('N')
#     ax.set_theta_direction(-1)
#     ax.set_thetagrids(range(0, 360, 45))
#     ax.set_title('Wind Rose')

#     plt.show()

# # Call the function to plot the wind rose
# plot_wind_rose(cleaned_df)

# Scatter plot of temperature vs. relative humidity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RH', y='Tamb', data=cleaned_df)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature vs. Relative Humidity')
plt.show()

# Combined plot of solar radiation and temperature vs. relative humidity
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='RH', y='GHI', data=cleaned_df)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Global Horizontal Irradiance (W/m²)')
plt.title('GHI vs. Relative Humidity')

plt.subplot(1, 2, 2)
sns.scatterplot(x='RH', y='Tamb', data=cleaned_df)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature vs. Relative Humidity')

plt.tight_layout()
plt.show()

# Histograms: Create histograms for variables like GHI, DNI, DHI, WS, and temperatures to visualize the frequency distribution of these variables.
# Histograms for GHI, DNI, DHI, WS, and Temperatures
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(data=cleaned_df, x='GHI', kde=True)
plt.title('Histogram of GHI')

plt.subplot(2, 3, 2)
sns.histplot(data=cleaned_df, x='DNI', kde=True)
plt.title('Histogram of DNI')

plt.subplot(2, 3, 3)
sns.histplot(data=cleaned_df, x='DHI', kde=True)
plt.title('Histogram of DHI')

plt.subplot(2, 3, 4)
sns.histplot(data=cleaned_df, x='WS', kde=True)
plt.title('Histogram of Wind Speed')

plt.subplot(2, 3, 5)
sns.histplot(data=cleaned_df, x='TModA', kde=True)
plt.title('Histogram of Module Temperature A')

plt.subplot(2, 3, 6)
sns.histplot(data=cleaned_df, x='TModB', kde=True)
plt.title('Histogram of Module Temperature B')

plt.tight_layout()
plt.show()

# Z-Score Analysis: Calculate Z-scores to flag data points that are significantly different from the mean
# Calculate Z-scores for numeric columns
numeric_cols = ['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'TModA', 'TModB', 'Tamb']
z_scores = (cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std()

# Identify outliers based on Z-score threshold (e.g., 3)
outliers = z_scores[z_scores.abs() > 3]

# Print or visualize outliers
print(outliers)
plt.figure(figsize=(12, 6))
sns.boxplot(data=z_scores)
plt.title('Z-Scores for Numeric Variables')
plt.show()

# Bubble charts to explore complex relationships between variables, such as GHI vs. Tamb vs. WS, with bubble size representing an additional variable like RH or BP (Barometric Pressure)

# Bubble plot of GHI vs. Tamb with bubble size representing RH
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GHI', y='Tamb', size='RH', hue='RH', data=cleaned_df, alpha=0.7)
plt.xlabel('Global Horizontal Irradiance (W/m²)')
plt.ylabel('Temperature (°C)')
plt.title('GHI vs. Tamb with RH as Bubble Size')
plt.show()

# Bubble plot of GHI vs. WS with bubble size representing BP
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GHI', y='WS', size='BP', hue='BP', data=cleaned_df, alpha=0.7)
plt.xlabel('Global Horizontal Irradiance (W/m²)')
plt.ylabel('Wind Speed (m/s)')
plt.title('GHI vs. WS with BP as Bubble Size')
plt.show()