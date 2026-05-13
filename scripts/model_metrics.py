from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def calculate_performance(y_true, y_pred):
    """
    Returns a dictionary of key fraud detection metrics.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": report['1']['recall'],
        "precision": report['1']['precision'],
        "f1_score": report['1']['f1-score']
    }

def print_matrix(y_true, y_pred):
    print("--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_true, y_pred))
