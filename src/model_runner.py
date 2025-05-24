import sys
sys.path.append('..')
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from src.analyse import InsuranceDataProcessor

class ModelRunner():
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Gaussian Naive Bayes': GaussianNB()
        }
    def _run_single_model(self, name, model, X_train, X_test, y_train, y_test, processor):
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score, report, cm = processor.analyze_result(name, y_test, y_pred)
        print(f"Training {name}... Done")
        return (name, score, report, cm)
    
    def run(self, X_train_scaled, X_test_scaled, y_train, y_test):
        nameList = []
        scoreList = []
        reportList = []
        cmList = []
        processor = InsuranceDataProcessor()
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._run_single_model, name, model,
                    X_train_scaled, X_test_scaled, y_train, y_test, processor
                )
                for name, model in self.models.items()
            ]

            for future in futures:
                name, score, report, cm = future.result()
                nameList.append(name)
                scoreList.append(score)
                reportList.append(report)
                cmList.append(cm)

        return nameList, scoreList, reportList, cm