import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the train.csv file
df = pd.read_csv('train.csv')

# Clean up the 'Total Assets' and 'Liabilities' columns
df['Total Assets'] = df['Total Assets'].apply(lambda x: re.sub(r'\D', '', str(x)))
df['Liabilities'] = df['Liabilities'].apply(lambda x: re.sub(r'\D', '', str(x)))

# Convert the cleaned columns to numeric data type
df['Total Assets'] = pd.to_numeric(df['Total Assets'], errors='coerce')
df['Liabilities'] = pd.to_numeric(df['Liabilities'], errors='coerce')

# Drop rows with NaN values in 'Total Assets' and 'Liabilities'
df.dropna(subset=['Total Assets', 'Liabilities'], inplace=True)

# Distinguishing Line Plot of Party vs Total Assets and Liabilities
party_stats = df.groupby('Party').agg({'Total Assets': 'mean', 'Liabilities': 'mean'})

plt.figure(figsize=(10,6))
plt.plot(party_stats['Total Assets'], label='Total Assets')
plt.plot(party_stats['Liabilities'], label='Liabilities')
plt.title('Party-wise Total Assets and Liabilities')
plt.xlabel('Party')
plt.ylabel('Amount')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# Filter the dataframe for candidates with criminal cases > 5
df_filtered = df[df['Criminal Case'] > 5]

# Count the number of candidates for each party
party_counts = df_filtered['Party'].value_counts()

# Calculate the percentage distribution
party_percentage = (party_counts / party_counts.sum()) * 100

# Plotting the pie chart
plt.figure(figsize=(10, 8))
plt.pie(party_percentage, labels=party_percentage.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage Distribution of Parties with Candidates having Criminal Cases > 5')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# Count the number of candidates for each predicted education level
education_counts = df['Education'].value_counts()

# Plotting the bar graph
plt.figure(figsize=(10, 6))
education_counts.plot(kind='bar', color='green')
plt.title('Predicted Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Count the number of candidates for each state
state_counts = df['state'].value_counts()

# Plotting the bar graph
plt.figure(figsize=(12, 6))
state_counts.plot(kind='bar', color='blue')
plt.title('Count of Members per State')
plt.xlabel('State')
plt.ylabel('Number of Members')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



