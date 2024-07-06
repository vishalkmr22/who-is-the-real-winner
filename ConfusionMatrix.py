import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming Y_val contains the true labels and val_predictions_rf contains the predicted labels
# Generate example data for demonstration
Y_val = [0, 1, 0, 1, 0, 1, 0, 1]
val_predictions_rf = [0, 1, 0, 1, 1, 1, 0, 0]

# Compute confusion matrix
cm_rf = confusion_matrix(Y_val, val_predictions_rf)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d")
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_rf.png')
plt.show()
