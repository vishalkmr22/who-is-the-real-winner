import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the training and test data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Define features and target
features = ['Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']
target = 'Education'

# Convert categorical variables to numeric
le = LabelEncoder()

# Combine the training and test data before fitting the LabelEncoder
combined_data = pd.concat([df_train[features], df_test[features]])

# Encode categorical features
for feature in features:
    combined_data[feature] = le.fit_transform(combined_data[feature])

# Split the combined data back into train and test
df_train[features] = combined_data[:len(df_train)]
df_test[features] = combined_data[len(df_train):]

# Prepare data for model
X_train = df_train[features]
Y_train = le.fit_transform(df_train[target])
X_test = df_test[features]

# Train the model
model = RandomForestClassifier(max_leaf_nodes=1000, random_state=2)
model.fit(X_train, Y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Convert numeric predictions back to original classes
predictions = le.inverse_transform(predictions)

# Write predictions to a CSV file
submission_df = pd.DataFrame({'ID': df_test['ID'], 'Education': predictions})
submission_df.to_csv('221201.csv', index=False)

# Plot distribution of each feature
for feature in features:
    plt.figure(figsize=(10, 6), dpi=120)
    
    # For 'Total Assets' and 'Liabilities', categorize values into ranges for better visibility
    if feature in ['Total Assets', 'Liabilities']:
        df_train[feature] = pd.cut(df_train[feature], bins=5)  # Adjust the number of bins as needed
        
    df_train[feature].value_counts().plot(kind='bar', color='skyblue', width=0.8)
    plt.title('Distribution of ' + feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plot distribution of predicted education values from the final CSV file
df_final = pd.read_csv('221201.csv')
plt.figure(figsize=(8, 6), dpi=120)
df_final['Education'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Predicted Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
