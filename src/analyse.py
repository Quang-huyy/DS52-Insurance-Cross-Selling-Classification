
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
class InsuranceDataProcessor:

    def analyze_dataset(self, df, title):
        stats = {
            'shape': df.shape,
            'head': df.head(),
            'info': df.info(),
            'describe': df.describe(),
            'null_counts': df.isnull().sum(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }

        print(f"\n{title} - Data overview:")
        print(df.head())
        
        print(f"\n{title} - Data types:")
        print(df.info())

        print(f"\n{title} - Data stats:")
        print(df.describe())

        print(f"\n{title} - Data missing values:")
        print(df.isnull().sum())

        print(f"\n{title} - Data unique values count:")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")

        return stats
    
    def analyze_result(self, model, y_test, y_pred):
        print(f"\n{model} - Evaluating")
        score = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return score, report, cm


