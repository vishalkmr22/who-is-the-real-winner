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


