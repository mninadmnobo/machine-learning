import numpy as np
import os
import time
from datetime import datetime
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTrees
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Set working directory to project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from extra_trees import ExtraTreesClassifier


def load_dataset(dataset_name):
    """Load Iris or Wine dataset"""
    if dataset_name.lower() == 'iris':
        dataset = load_iris()
    elif dataset_name.lower() == 'wine':
        dataset = load_wine()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset.data, dataset.target


def split_data(features, labels, test_size=0.3, random_state=42):
    """Split data into train-test sets"""
    return train_test_split(features, labels, test_size=test_size, 
                           random_state=random_state, stratify=labels)


def calculate_accuracy(true_labels, predicted_labels):
    """Calculate accuracy"""
    return np.mean(true_labels == predicted_labels)


def calculate_f1_score(true_labels, predicted_labels):
    """Calculate weighted F1-score"""
    unique_classes = np.unique(true_labels)
    f1_scores = []
    weights = []
    
    for class_label in unique_classes:
        true_binary = (true_labels == class_label).astype(int)
        pred_binary = (predicted_labels == class_label).astype(int)
        
        tp = np.sum((true_binary == 1) & (pred_binary == 1))
        fp = np.sum((true_binary == 0) & (pred_binary == 1))
        fn = np.sum((true_binary == 1) & (pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        weights.append(np.sum(true_labels == class_label))
    
    return np.sum(np.array(f1_scores) * np.array(weights)) / np.sum(weights)


def calculate_auroc(true_labels, predicted_probabilities):
    """Calculate weighted AUROC (Area Under ROC Curve)"""
    unique_classes = np.unique(true_labels)
    
    if len(predicted_probabilities.shape) == 1:
        predicted_probabilities = np.column_stack([1 - predicted_probabilities, predicted_probabilities])
    
    if len(unique_classes) == 2:
        try:
            return roc_auc_score(true_labels, predicted_probabilities[:, 1])
        except:
            return None
    
    try:
        binary_labels = label_binarize(true_labels, classes=unique_classes)
        return roc_auc_score(binary_labels, predicted_probabilities, 
                           average='weighted', multi_class='ovr')
    except:
        return None


def evaluate_model(model, train_features, test_features, train_labels, test_labels):
    """Evaluate model on train and test data"""
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    
    results = {
        'train_accuracy': calculate_accuracy(train_labels, train_predictions),
        'train_f1_score': calculate_f1_score(train_labels, train_predictions),
        'test_accuracy': calculate_accuracy(test_labels, test_predictions),
        'test_f1_score': calculate_f1_score(test_labels, test_predictions),
        'train_auroc': None,
        'test_auroc': None
    }
    
    # Try custom models (predict_probability) first, then sklearn (predict_proba)
    try:
        train_probs = model.predict_probability(train_features)
        results['train_auroc'] = calculate_auroc(train_labels, train_probs)
    except AttributeError:
        try:
            train_probs = model.predict_proba(train_features)
            results['train_auroc'] = calculate_auroc(train_labels, train_probs)
        except:
            pass
    except:
        pass
    
    try:
        test_probs = model.predict_probability(test_features)
        results['test_auroc'] = calculate_auroc(test_labels, test_probs)
    except AttributeError:
        try:
            test_probs = model.predict_proba(test_features)
            results['test_auroc'] = calculate_auroc(test_labels, test_probs)
        except:
            pass
    except:
        pass
    
    return results


def format_results_table(results_dict):
    """Format results as a printable table."""
    output = []
    output.append("-" * 90)
    output.append(f"{'Model':<30} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12}")
    output.append("-" * 90)
    
    for model_name, metrics in results_dict.items():
        train_acc = f"{metrics['train_accuracy']*100:.2f}%"
        test_acc = f"{metrics['test_accuracy']*100:.2f}%"
        train_f1 = f"{metrics['train_f1_score']:.4f}"
        test_f1 = f"{metrics['test_f1_score']:.4f}"
        output.append(f"{model_name:<30} {train_acc:<12} {test_acc:<12} {train_f1:<12} {test_f1:<12}")
    
    output.append("-" * 90)
    return "\n".join(output)


def run_experiment(dataset_name, random_state=42):
    """Run experiment on a single dataset"""
    features, labels = load_dataset(dataset_name)
    train_features, test_features, train_labels, test_labels = split_data(
        features, labels, test_size=0.3, random_state=random_state
    )
    
    results = {}
    execution_times = {}
    
    algorithms = [
        ('Decision Tree (Custom)', DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=random_state)),
        ('Random Forest (Custom)', RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', random_state=random_state)),
        ('Extra Trees (Custom)', ExtraTreesClassifier(n_estimators=100, max_depth=10, max_features='sqrt', random_state=random_state)),
        ('Decision Tree (Sklearn)', SklearnDecisionTree(max_depth=10, min_samples_split=2, random_state=random_state)),
        ('Random Forest (Sklearn)', SklearnRandomForest(n_estimators=100, max_depth=10, max_features='sqrt', random_state=random_state)),
        ('Extra Trees (Sklearn)', SklearnExtraTrees(n_estimators=100, max_depth=10, max_features='sqrt', random_state=random_state)),
    ]
    
    for algo_name, model in algorithms:
        start = time.time()
        model.fit(train_features, train_labels)
        exec_time = time.time() - start
        
        eval_results = evaluate_model(model, train_features, test_features, train_labels, test_labels)
        results[algo_name] = eval_results
        execution_times[algo_name] = exec_time
    
    total_time = sum(execution_times.values())
    
    return results, total_time, len(train_features), len(test_features), features.shape[1]


def save_combined_results(all_results):
    """Save combined results for all datasets to a single file"""
    os.makedirs('results', exist_ok=True)
    
    filename = 'results/results.txt'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'w') as f:
        f.write("=" * 130 + "\n")
        f.write("TREE ENSEMBLE EXPERIMENTS - COMBINED RESULTS\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 130 + "\n\n")
        
        for dataset_name, data in all_results.items():
            results_dict = data['results']
            train_size = data['train_size']
            test_size = data['test_size']
            n_features = data['n_features']
            exec_time = data['exec_time']
            
            f.write(f"{dataset_name.upper()} DATASET\n")
            f.write(f"Data: {train_size + test_size} samples ({train_size} train, {test_size} test) | Features: {n_features} | Time: {exec_time:.2f}s\n")
            f.write("-" * 130 + "\n")
            f.write(f"{'Algorithm':<30} {'Train Acc':<12} {'Test Acc':<12} {'Train F1':<12} {'Test F1':<12} {'Train AUROC':<12} {'Test AUROC':<12}\n")
            f.write("-" * 130 + "\n")
            
            for algorithm, metrics in results_dict.items():
                train_acc = metrics['train_accuracy'] * 100
                test_acc = metrics['test_accuracy'] * 100
                train_f1 = metrics['train_f1_score']
                test_f1 = metrics['test_f1_score']
                train_auroc = metrics['train_auroc']
                test_auroc = metrics['test_auroc']
                
                train_auroc_str = f"{train_auroc:.4f}" if train_auroc is not None else "N/A"
                test_auroc_str = f"{test_auroc:.4f}" if test_auroc is not None else "N/A"
                
                f.write(f"{algorithm:<30} {train_acc:11.2f}% {test_acc:11.2f}% {train_f1:11.4f} {test_f1:11.4f} {train_auroc_str:>11} {test_auroc_str:>11}\n")
            
            f.write("=" * 130 + "\n\n")


def main():
    """Main entry point"""
    all_results = {}
    
    for dataset in ['iris', 'wine']:
        results, exec_time, train_size, test_size, n_features = run_experiment(dataset, random_state=42)
        all_results[dataset] = {
            'results': results,
            'exec_time': exec_time,
            'train_size': train_size,
            'test_size': test_size,
            'n_features': n_features
        }
    
    save_combined_results(all_results)


if __name__ == "__main__":
    main()
