# Machine Learning Model for Education Prediction

This project implements a machine learning model to predict the education level of state election winners based on various features such as party affiliation, criminal cases, total assets, liabilities, and state.

## Dataset

The dataset consists of two CSV files: `train.csv` and `test.csv`. The training dataset (`train.csv`) contains features and the target variable (education level) for training the model, while the test dataset (`test.csv`) contains only features for making predictions.

## Dependencies

The following Python libraries are required to run the code:
- matplotlib
- pandas
- scikit-learn

Install the dependencies using the following command:
pip install matplotlib pandas scikit-learn


## Usage

1. Clone the repository:
git clone <repository-url>
cd <repository-name>


2. Run the Python script `221201.py`:
python 221201.py

This script trains a RandomForestClassifier model on the training dataset, makes predictions on the test dataset, and generates a CSV file (`221201.csv`) containing the predicted education levels.

## Model Details

The RandomForestClassifier model is trained with the following hyperparameters:
- Max Leaf Nodes: 1000
- Random State: 2

## Results

The final predictions are visualized using bar plots to show the distribution of each feature and the predicted education levels.

## License

This project is licensed under the IITK License.

