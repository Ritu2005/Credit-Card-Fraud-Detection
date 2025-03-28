# Credit Card Fraud Detection

## Task Objectives
- Build a machine learning model to identify fraudulent credit card transactions.
- Perform data preprocessing, including feature engineering and exploratory data analysis.
- Evaluate and compare the model's accuracy using logistic regression and random forest.
- Minimize false positives while maximizing fraud detection accuracy.

## Steps to Run the Project

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Install Dependencies**
Make sure Python and pip are installed. Then run:
```bash
pip install -r requirements.txt
```

3. **Add Dataset**
- Download the dataset and place it in a directory named `input/fraud-detection/`.
- Ensure the dataset files are named `fraudTrain.csv` and `fraudTest.csv`.

4. **Run the Code**
```bash
python fraud_detection.py
```

5. **Output**
- The model accuracy, confusion matrix, and classification report will be displayed in the console.
- Visualization plots will show data distributions and transaction patterns.

## Project Structure
```
credit-card-fraud-detection/
│
├── fraud_detection.py   # Main script for data analysis and model building
├── requirements.txt     # List of required packages
└── input/
    └── fraud-detection/
        ├── fraudTrain.csv
        ├── fraudTest.csv
```

## Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

## Notes
- Ensure the data paths are correct.
- The script uses both Logistic Regression and Random Forest models for classification.
- Adjust hyperparameters if necessary for better results.

## License
This project is licensed under the MIT License. Feel free to contribute and improve!

