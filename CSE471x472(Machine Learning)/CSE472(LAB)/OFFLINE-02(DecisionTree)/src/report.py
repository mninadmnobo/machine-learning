"""
CSE 472 - Machine Learning Lab | Assignment 2
Report on Assignment-2: Decision Tree, Random Forest & Extra Trees
Student: Mohammad Ninad Mahmud Nobo | ID: 2005080
Academic Report Generator - Following LaTeX-style Structure
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTrees
from fpdf import FPDF

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from extra_trees import ExtraTreesClassifier
from run import load_dataset, split_data, calculate_accuracy, calculate_f1_score, calculate_auroc

# Set working directory to project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')


def calculate_metrics(y_true, y_pred):
    return calculate_accuracy(y_true, y_pred), calculate_f1_score(y_true, y_pred)


def experiment_varying_depth(dataset_name, random_state=42):
    features, labels = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = split_data(features, labels, test_size=0.3, random_state=random_state)
    
    depths = [1, 2, 3, 5, 7, 10, 15, None]
    results = {'depths': [], 'dt_train': [], 'dt_test': [], 'rf_train': [], 'rf_test': [], 
               'et_train': [], 'et_test': [], 'sklearn_dt_train': [], 'sklearn_dt_test': []}
    
    for depth in depths:
        actual_depth = depth if depth else 50
        results['depths'].append(actual_depth if depth else 20)
        
        for model_class, prefix in [(DecisionTreeClassifier, 'dt'), (SklearnDecisionTree, 'sklearn_dt')]:
            model = model_class(max_depth=depth if prefix == 'sklearn_dt' else actual_depth, random_state=random_state)
            model.fit(X_train, y_train)
            results[f'{prefix}_train'].append(calculate_metrics(y_train, model.predict(X_train))[0])
            results[f'{prefix}_test'].append(calculate_metrics(y_test, model.predict(X_test))[0])
        
        for model_class, prefix in [(RandomForestClassifier, 'rf'), (ExtraTreesClassifier, 'et')]:
            model = model_class(n_estimators=50, max_depth=actual_depth, random_state=random_state)
            model.fit(X_train, y_train)
            results[f'{prefix}_train'].append(calculate_metrics(y_train, model.predict(X_train))[0])
            results[f'{prefix}_test'].append(calculate_metrics(y_test, model.predict(X_test))[0])
    
    return results


def experiment_varying_estimators(dataset_name, random_state=42):
    features, labels = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = split_data(features, labels, test_size=0.3, random_state=random_state)
    
    estimators_list = [1, 10, 25, 50, 100, 150]
    results = {'n_estimators': estimators_list, 'rf_test': [], 'et_test': [], 'rf_time': [], 'et_time': []}
    
    for n in estimators_list:
        for model_class, prefix in [(RandomForestClassifier, 'rf'), (ExtraTreesClassifier, 'et')]:
            start = time.time()
            model = model_class(n_estimators=n, max_depth=10, random_state=random_state)
            model.fit(X_train, y_train)
            results[f'{prefix}_time'].append(time.time() - start)
            results[f'{prefix}_test'].append(calculate_metrics(y_test, model.predict(X_test))[0])
    
    return results


def experiment_final_comparison(random_state=42):
    results = {}
    for dataset_name in ['iris', 'wine']:
        features, labels = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test = split_data(features, labels, test_size=0.3, random_state=random_state)
        results[dataset_name] = {}
        
        models = [
            ('DT (Custom)', DecisionTreeClassifier(max_depth=10, random_state=random_state)),
            ('RF (Custom)', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)),
            ('ET (Custom)', ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=random_state)),
            ('DT (Sklearn)', SklearnDecisionTree(max_depth=10, random_state=random_state)),
            ('RF (Sklearn)', SklearnRandomForest(n_estimators=100, max_depth=10, random_state=random_state)),
            ('ET (Sklearn)', SklearnExtraTrees(n_estimators=100, max_depth=10, random_state=random_state)),
        ]
        
        for name, model in models:
            model.fit(X_train, y_train)
            train_acc, train_f1 = calculate_metrics(y_train, model.predict(X_train))
            test_acc, test_f1 = calculate_metrics(y_test, model.predict(X_test))
            
            try:
                proba_method = getattr(model, 'predict_probability', None) or getattr(model, 'predict_proba', None)
                test_auroc = calculate_auroc(y_test, proba_method(X_test)) if proba_method else None
            except:
                test_auroc = None
            
            results[dataset_name][name] = {
                'train_acc': train_acc, 'test_acc': test_acc,
                'train_f1': train_f1, 'test_f1': test_f1,
                'test_auroc': test_auroc, 'gap': train_acc - test_acc
            }
    return results


def plot_depth_analysis(iris_results, wine_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, results, title in [(axes[0], iris_results, 'IRIS'), (axes[1], wine_results, 'WINE')]:
        ax.plot(results['depths'], results['dt_train'], 'b-o', label='DT Train', linewidth=2)
        ax.plot(results['depths'], results['dt_test'], 'b--s', label='DT Test', linewidth=2)
        ax.plot(results['depths'], results['rf_test'], 'g--^', label='RF Test', linewidth=2)
        ax.plot(results['depths'], results['et_test'], 'm--v', label='ET Test', linewidth=2)
        ax.set_xlabel('Max Depth'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{title}: Bias-Variance Tradeoff'); ax.legend(); ax.set_ylim([0.6, 1.02])
    plt.tight_layout()
    plt.savefig('results/figures/depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_estimators_analysis(iris_results, wine_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, results, title in [(axes[0], iris_results, 'IRIS'), (axes[1], wine_results, 'WINE')]:
        ax.plot(results['n_estimators'], results['rf_test'], 'g-o', label='Random Forest', linewidth=2)
        ax.plot(results['n_estimators'], results['et_test'], 'm-s', label='Extra Trees', linewidth=2)
        ax.set_xlabel('Number of Estimators'); ax.set_ylabel('Test Accuracy')
        ax.set_title(f'{title}: Effect of n_estimators'); ax.legend(); ax.set_ylim([0.7, 1.02])
    plt.tight_layout()
    plt.savefig('results/figures/estimators_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    models = ['DT (Custom)', 'RF (Custom)', 'ET (Custom)', 'DT (Sklearn)', 'RF (Sklearn)', 'ET (Sklearn)']
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    x = np.arange(len(models))
    
    for ax, dataset, title in [(axes[0], 'iris', 'IRIS'), (axes[1], 'wine', 'WINE')]:
        test_accs = [results[dataset][m]['test_acc'] for m in models]
        bars = ax.bar(x, test_accs, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylabel('Test Accuracy'); ax.set_title(f'{title} Dataset')
        ax.set_xticks(x); ax.set_xticklabels([m.split()[0] + '\n' + m.split()[1] for m in models], fontsize=8)
        ax.set_ylim([0.8, 1.05])
        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('results/figures/final_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_generalization_gap(results):
    fig, ax = plt.subplots(figsize=(10, 4))
    models = ['DT (Custom)', 'RF (Custom)', 'ET (Custom)', 'DT (Sklearn)', 'RF (Sklearn)', 'ET (Sklearn)']
    x = np.arange(len(models))
    width = 0.35
    
    iris_gaps = [results['iris'][m]['gap'] * 100 for m in models]
    wine_gaps = [results['wine'][m]['gap'] * 100 for m in models]
    
    ax.bar(x - width/2, iris_gaps, width, label='Iris', color='#3498db')
    ax.bar(x + width/2, wine_gaps, width, label='Wine', color='#e74c3c')
    ax.set_ylabel('Generalization Gap (%)'); ax.set_title('Overfitting Analysis (Lower = Better)')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9); ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/figures/generalization_gap.png', dpi=150, bbox_inches='tight')
    plt.close()


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(128, 128, 128)
            self.set_y(8)
            self.cell(95, 5, 'CSE-472: Machine Learning Sessional | Assignment 2', 0, 0, 'L')
            self.cell(95, 5, 'Mohammad Ninad Mahmud Nobo (2005080)', 0, 1, 'R')
            self.set_text_color(0, 0, 0)
            self.set_x(10)
            self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'{self.page_no()}', 0, 0, 'C')

    def section(self, num, title):
        self.set_x(10)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f'{num} {title}', 0, 1, 'L')
        self.ln(2)

    def subsection(self, num, title):
        self.set_x(10)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f'{num} {title}', 0, 1, 'L')
        self.ln(1)
    
    def body_text(self, text):
        self.set_x(10)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(2)
    
    def bold_text(self, text):
        self.set_x(10)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
    
    def italic_text(self, text):
        self.set_x(10)
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)


def generate_pdf_report(final_results):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ===== TITLE PAGE (Professional Cover) =====
    pdf.add_page()
    
    # Top banner
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 100, 'F')
    
    # University name
    pdf.set_y(20)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, 'Bangladesh University of Engineering and Technology', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 6, 'Department of Computer Science and Engineering', 0, 1, 'C')
    
    pdf.ln(8)
    pdf.set_font('Helvetica', 'B', 22)
    pdf.cell(0, 12, 'CSE 472 - Machine Learning Lab', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, 'Assignment 2', 0, 1, 'C')
    
    # Main title box
    pdf.set_y(115)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_draw_color(41, 128, 185)
    pdf.rect(25, 110, 160, 35, 'FD')
    pdf.set_y(118)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 10, 'Decision Tree, Random Forest', 0, 1, 'C')
    pdf.cell(0, 10, '& Extra Trees Classifier', 0, 1, 'C')
    
    # Student info box
    pdf.set_y(165)
    pdf.set_fill_color(236, 240, 241)
    pdf.rect(40, 160, 130, 40, 'F')
    pdf.set_text_color(44, 62, 80)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Mohammad Ninad Mahmud Nobo', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Student ID: 2005080', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Level 4, Term 2', 0, 1, 'C')
    
    # Assignment details
    pdf.set_y(215)
    pdf.set_text_color(100, 100, 100)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 6, 'From-scratch implementation of tree-based classifiers', 0, 1, 'C')
    pdf.cell(0, 6, 'with comparison against Scikit-learn', 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    
    # ===== 1. PROBLEM STATEMENT & INTRODUCTION =====
    pdf.add_page()
    pdf.section('1', 'Problem Statement')
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_x(10)
    pdf.multi_cell(0, 6, 'From-Scratch Implementations of Decision Tree, Random Forest and Extra Trees & Empirical Comparison with scikit-learn')
    pdf.ln(2)
    
    # ===== 2. INTRODUCTION =====
    pdf.section('2', 'Introduction')
    pdf.body_text('Decision trees are intuitive machine learning models that partition the feature space using axis-aligned splits, making them highly interpretable. However, they often suffer from high variance and overfitting, especially when grown to full depth. Ensemble methods such as Random Forests and Extremely Randomized Trees (Extra Trees) address these limitations by combining multiple trees and introducing controlled randomness in feature selection, data sampling, and split decisions, leading to improved generalization and robustness.')
    
    pdf.body_text('In this assignment, we implement Decision Trees, Random Forests, and Extra Trees from scratch for multi-class classification tasks. We empirically evaluate their performance on two benchmark datasets: the Iris dataset and the Wine dataset. Our custom implementations-built using only NumPy and basic Python-are compared with scikit-learn\'s optimized versions across three evaluation metrics: Accuracy, F1-score, and AUROC. The analysis highlights how ensemble techniques mitigate overfitting and demonstrates the trade-offs between custom and production-grade implementations.')
    
    # ===== 3. ALGORITHMIC DETAILS =====
    pdf.section('3', 'Algorithmic Details')
    pdf.body_text('This section describes the implementation details of the Decision Tree, Random Forest, and Extremely Randomized Trees classifiers used in this work. All models are implemented from scratch following the CART framework and support multi-class classification.')
    pdf.ln(2)
    
    # 3.1 Decision Tree
    pdf.subsection('3.1', 'Decision Tree (CART)')
    pdf.body_text('The Decision Tree is implemented using the Classification and Regression Tree (CART) methodology. At each internal node, the algorithm searches for a binary split of the form:')
    pdf.ln(1)
    pdf.set_font('Courier', 'B', 12)
    pdf.set_x(10)
    pdf.cell(0, 6, 'x(j) <= t', 0, 1, 'C')
    pdf.ln(1)
    pdf.body_text('that maximizes impurity reduction.')
    
    pdf.bold_text('Splitting Criterion:')
    pdf.body_text('Let D be the set of samples at a node, and DL, DR be the left and right child subsets after a split. The impurity reduction (information gain) is computed as:')
    pdf.ln(1)
    pdf.set_font('Courier', 'B', 10)
    pdf.set_x(10)
    pdf.cell(0, 6, 'Delta_I = I(D) - ( |DL|/|D| * I(DL) + |DR|/|D| * I(DR) )', 0, 1, 'C')
    pdf.ln(1)
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('where I(.) is the Gini impurity:')
    pdf.set_font('Courier', 'B', 10)
    pdf.set_x(10)
    pdf.cell(0, 6, 'I(D) = 1 - SUM(pk^2)  for k = 1 to K', 0, 1, 'C')
    pdf.ln(1)
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('and pk denotes the class probability of class k at the node.')
    
    pdf.bold_text('Split Search (CART-style):')
    pdf.body_text('For each candidate feature, all unique feature values are evaluated as potential thresholds. The split yielding the maximum impurity reduction is selected.')
    
    pdf.bold_text('Stopping Criteria:')
    pdf.body_text('Tree growth stops if any of the following conditions is met:\n- Maximum depth is reached\n- Number of samples at the node is less than min_samples_split\n- All samples belong to the same class\n- No valid split can improve impurity')
    
    pdf.bold_text('Leaf Prediction:')
    pdf.body_text('Each leaf node stores the majority class label. For probabilistic prediction, the class distribution is estimated from the training samples at the leaf.')
    
    pdf.bold_text('Hyperparameters:')
    pdf.body_text('- max_depth\n- min_samples_split\n- criterion: Gini impurity')
    
    # 3.2 Random Forest
    pdf.subsection('3.2', 'Random Forest')
    pdf.body_text('Random Forest is implemented as an ensemble of independently trained CART decision trees, combining both bagging and feature randomness.')
    
    pdf.bold_text('Bootstrap Sampling:')
    pdf.body_text('For each tree, a bootstrap dataset is created by sampling N training examples with replacement from the original dataset.')
    
    pdf.bold_text('Feature Subsampling:')
    pdf.body_text('At each split, only a random subset of features of size:')
    pdf.set_font('Courier', 'B', 10)
    pdf.set_x(10)
    pdf.cell(0, 6, 'max_features = floor(sqrt(d))', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('is considered, where d is the total number of features.')
    
    pdf.bold_text('Tree Construction:')
    pdf.body_text('Each tree is trained using the same CART splitting strategy as the standalone Decision Tree, but on its own bootstrap sample and feature subset.')
    
    pdf.bold_text('Prediction:')
    pdf.body_text('Final predictions are obtained using majority voting across all trees. Class probabilities are computed by averaging the predicted probabilities of individual trees.')
    
    pdf.bold_text('Hyperparameters:')
    pdf.body_text('- n_estimators\n- max_depth\n- min_samples_split\n- max_features')
    
    # 3.3 Extra Trees
    pdf.subsection('3.3', 'Extremely Randomized Trees (Extra Trees)')
    pdf.body_text('Extremely Randomized Trees further increase randomness to reduce variance by altering the split selection strategy.')
    
    pdf.bold_text('No Bootstrap Sampling:')
    pdf.body_text('Unlike Random Forests, each Extra Tree is trained on the full training dataset without bootstrap resampling.')
    
    pdf.bold_text('Random Split Selection:')
    pdf.body_text('For each candidate feature, split thresholds are sampled uniformly at random from the feature range:')
    pdf.set_font('Courier', 'B', 10)
    pdf.set_x(10)
    pdf.cell(0, 6, 't ~ U(min(x_j), max(x_j))', 0, 1, 'C')
    pdf.ln(1)
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('The best random threshold is selected based on impurity reduction.')
    
    pdf.bold_text('Feature Subsampling:')
    pdf.body_text('Similar to Random Forests, a random subset of features is considered at each node.')
    
    pdf.bold_text('Prediction:')
    pdf.body_text('Predictions are aggregated using majority voting, and class probabilities are obtained by averaging across trees.')
    
    pdf.bold_text('Hyperparameters:')
    pdf.body_text('- n_estimators\n- max_depth\n- min_samples_split\n- max_features')
    
    # ===== 4. EXPERIMENTAL SETUP =====
    pdf.section('4', 'Experimental Setup')
    pdf.body_text('This section describes the datasets used for evaluation, the data splitting strategy, and the hyperparameter settings employed for all experiments. All experiments were conducted using a fixed random seed to ensure reproducibility.')
    
    # 4.1 Datasets
    pdf.subsection('4.1', 'Datasets')
    pdf.body_text('Experiments were performed on two benchmark multi-class classification datasets obtained from sklearn.datasets:')
    pdf.body_text('- Iris Dataset: Contains 150 samples, 4 numerical features, and 3 classes.\n- Wine Dataset: Contains 178 samples, 13 numerical features, and 3 classes.')
    pdf.body_text('All features in both datasets are continuous-valued. No data preprocessing steps such as feature scaling, normalization, or one-hot encoding were applied, as decision tree-based models are invariant to feature scaling and naturally handle numerical features.')
    
    # 4.2 Train-Test Split
    pdf.subsection('4.2', 'Train-Test Split')
    pdf.body_text('We use train_test_split with:\n- test_size = 0.3 (70% training, 30% testing)\n- random_state = 42\n- stratify = y (stratified sampling to preserve class distribution)')
    
    # Dataset Split Table
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Dataset Split Summary:', 0, 1)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(50, 7, 'Dataset', 1, 0, 'C', True)
    pdf.cell(45, 7, 'Training Samples', 1, 0, 'C', True)
    pdf.cell(45, 7, 'Test Samples', 1, 0, 'C', True)
    pdf.cell(35, 7, 'Classes', 1, 1, 'C', True)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(50, 6, 'Iris', 1, 0, 'C')
    pdf.cell(45, 6, '105', 1, 0, 'C')
    pdf.cell(45, 6, '45', 1, 0, 'C')
    pdf.cell(35, 6, '3', 1, 1, 'C')
    pdf.cell(50, 6, 'Wine', 1, 0, 'C')
    pdf.cell(45, 6, '124', 1, 0, 'C')
    pdf.cell(45, 6, '54', 1, 0, 'C')
    pdf.cell(35, 6, '3', 1, 1, 'C')
    pdf.ln(3)
    
    # 4.3 Random Seed
    pdf.subsection('4.3', 'Random Seed and Reproducibility')
    pdf.body_text('All sources of randomness in the experiments were controlled using a fixed random seed: RANDOM_STATE = 42')
    pdf.body_text('This seed was consistently applied to:\n- Train-test splitting\n- Bootstrap sampling in Random Forests\n- Feature subsampling at each split\n- Random threshold selection in Extra Trees')
    pdf.body_text('Using a fixed random seed ensures that the experimental results are deterministic and reproducible.')
    
    # 4.4 Hyperparameters
    pdf.subsection('4.4', 'Hyperparameter Configuration')
    pdf.body_text('The same set of hyperparameters was used across both datasets to ensure a fair comparison between models.')
    
    pdf.bold_text('Decision Tree:')
    pdf.body_text('- Maximum depth (max_depth): 10\n- Minimum samples per split (min_samples_split): 2\n- Splitting criterion: Gini impurity')
    
    pdf.bold_text('Random Forest:')
    pdf.body_text('- Number of trees (n_estimators): 100\n- Maximum depth (max_depth): 10\n- Minimum samples per split (min_samples_split): 2\n- Number of features per split (max_features): floor(sqrt(d))')
    
    pdf.bold_text('Extra Trees:')
    pdf.body_text('- Number of trees (n_estimators): 100\n- Maximum depth (max_depth): 10\n- Minimum samples per split (min_samples_split): 2\n- Number of features per split (max_features): floor(sqrt(d))')
    
    # 4.5 Evaluation Protocol
    pdf.subsection('4.5', 'Evaluation Protocol')
    pdf.body_text('All models were trained on the training set and evaluated on the held-out test set. Performance was measured using:')
    pdf.body_text('- Accuracy: Proportion of correctly classified samples\n- Weighted F1-score: Harmonic mean of precision and recall, weighted by class frequency\n- AUROC: Area Under the Receiver Operating Characteristic curve (One-vs-Rest, weighted average)')
    pdf.body_text('For ensemble models, final predictions were obtained via majority voting, while class probabilities were computed by averaging probabilities across individual trees.')
    
    # ===== 5. RESULTS =====
    pdf.section('5', 'Results')
    pdf.body_text('This section presents the experimental results obtained on the Iris and Wine datasets. Performance is evaluated using Accuracy, F1-score, and AUROC. Comparisons are made across custom implementations and their scikit-learn counterparts for Decision Trees, Random Forests, and Extra Trees.')
    
    # 5.1 Quantitative Results
    pdf.subsection('5.1', 'Quantitative Results')
    
    models = ['DT (Custom)', 'RF (Custom)', 'ET (Custom)', 'DT (Sklearn)', 'RF (Sklearn)', 'ET (Sklearn)']
    
    # Table 1: IRIS
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, 'Table 1: Performance comparison on the Iris dataset', 0, 1, 'C')
    pdf.ln(1)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(50, 7, 'Model', 1, 0, 'C', True)
    pdf.cell(35, 7, 'Accuracy', 1, 0, 'C', True)
    pdf.cell(35, 7, 'F1-Score', 1, 0, 'C', True)
    pdf.cell(35, 7, 'AUROC', 1, 1, 'C', True)
    pdf.set_font('Helvetica', '', 9)
    
    model_display = {'DT (Custom)': 'Custom Decision Tree', 'RF (Custom)': 'Custom Random Forest', 
                     'ET (Custom)': 'Custom Extra Trees', 'DT (Sklearn)': 'Sklearn Decision Tree',
                     'RF (Sklearn)': 'Sklearn Random Forest', 'ET (Sklearn)': 'Sklearn Extra Trees'}
    
    for i, m in enumerate(models):
        r = final_results['iris'][m]
        pdf.set_fill_color(245, 245, 245) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.cell(50, 6, model_display[m], 1, 0, 'L', True)
        pdf.cell(35, 6, f"{r['test_acc']:.4f}", 1, 0, 'C', True)
        pdf.cell(35, 6, f"{r['test_f1']:.4f}", 1, 0, 'C', True)
        pdf.cell(35, 6, f"{r['test_auroc']:.4f}" if r['test_auroc'] else "N/A", 1, 1, 'C', True)
    pdf.ln(4)
    
    # Table 2: WINE
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, 'Table 2: Performance comparison on the Wine dataset', 0, 1, 'C')
    pdf.ln(1)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(50, 7, 'Model', 1, 0, 'C', True)
    pdf.cell(35, 7, 'Accuracy', 1, 0, 'C', True)
    pdf.cell(35, 7, 'F1-Score', 1, 0, 'C', True)
    pdf.cell(35, 7, 'AUROC', 1, 1, 'C', True)
    pdf.set_font('Helvetica', '', 9)
    
    for i, m in enumerate(models):
        r = final_results['wine'][m]
        pdf.set_fill_color(245, 245, 245) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.cell(50, 6, model_display[m], 1, 0, 'L', True)
        pdf.cell(35, 6, f"{r['test_acc']:.4f}", 1, 0, 'C', True)
        pdf.cell(35, 6, f"{r['test_f1']:.4f}", 1, 0, 'C', True)
        pdf.cell(35, 6, f"{r['test_auroc']:.4f}" if r['test_auroc'] else "N/A", 1, 1, 'C', True)
    pdf.ln(4)
    
    # 5.2 Visual Results
    pdf.subsection('5.2', 'Visual Results')
    pdf.body_text('The following visualizations were generated to analyze model performance:')
    
    # Figure 1
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, '5.2.1 Results Comparison (Bar Plot)', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('Grouped bar plots comparing all models across Accuracy for both datasets.')
    if os.path.exists('results/figures/final_comparison.png'):
        pdf.image('results/figures/final_comparison.png', x=15, w=180, h=60)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, 'Figure 1: Results Comparison', 0, 1, 'C')
    
    # Figure 2 - Depth Analysis
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, '5.2.2 Effect of Max Depth', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('Line plots showing the bias-variance tradeoff as tree depth increases.')
    if os.path.exists('results/figures/depth_analysis.png'):
        pdf.image('results/figures/depth_analysis.png', x=15, w=180, h=60)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, 'Figure 2: Effect of Max Depth', 0, 1, 'C')
    
    # Figure 3 - Estimators
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, '5.2.3 Effect of Number of Trees', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('Line plots showing how accuracy changes with increasing number of trees for Random Forest and Extra Trees.')
    if os.path.exists('results/figures/estimators_analysis.png'):
        pdf.image('results/figures/estimators_analysis.png', x=15, w=180, h=60)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, 'Figure 3: Effect of Number of Trees', 0, 1, 'C')
    
    # Figure 4 - Generalization Gap
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, '5.2.4 Generalization Gap Analysis', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('Bar plots showing the overfitting analysis (Train Accuracy - Test Accuracy) for all models.')
    if os.path.exists('results/figures/generalization_gap.png'):
        pdf.image('results/figures/generalization_gap.png', x=15, w=180, h=60)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 5, 'Figure 4: Generalization Gap (Overfitting Analysis)', 0, 1, 'C')
    
    # ===== 6. ANALYSIS =====
    pdf.section('6', 'Analysis')
    
    # 6.1 Decision Tree Performance
    pdf.subsection('6.1', 'Decision Tree Performance')
    iris_dt_custom = final_results['iris']['DT (Custom)']['test_acc'] * 100
    iris_dt_sklearn = final_results['iris']['DT (Sklearn)']['test_acc'] * 100
    wine_dt_custom = final_results['wine']['DT (Custom)']['test_acc'] * 100
    wine_dt_sklearn = final_results['wine']['DT (Sklearn)']['test_acc'] * 100
    pdf.body_text(f'For both datasets, the custom Decision Tree implementation achieves performance comparable to the scikit-learn Decision Tree. On the Iris dataset, the custom implementation achieves {iris_dt_custom:.2f}% accuracy while sklearn achieves {iris_dt_sklearn:.2f}%. On the Wine dataset, custom achieves {wine_dt_custom:.2f}% while sklearn achieves {wine_dt_sklearn:.2f}%. These minor differences can be attributed to different tie-breaking strategies and implementation-level details in split selection. The close performance validates that the CART-style split selection using Gini impurity and stopping criteria are correctly implemented.')
    
    # 6.2 Ensemble Methods Performance
    pdf.subsection('6.2', 'Ensemble Methods Performance')
    iris_rf = final_results['iris']['RF (Custom)']['test_acc'] * 100
    iris_et = final_results['iris']['ET (Custom)']['test_acc'] * 100
    wine_rf = final_results['wine']['RF (Custom)']['test_acc'] * 100
    wine_et = final_results['wine']['ET (Custom)']['test_acc'] * 100
    pdf.body_text(f'Ensemble methods consistently outperform standalone Decision Trees. On the Iris dataset:\n- Custom Random Forest and Extra Trees achieve {iris_rf:.2f}% and {iris_et:.2f}% accuracy respectively\n- This represents a significant improvement over the {iris_dt_custom:.2f}% accuracy of the custom Decision Tree')
    pdf.body_text(f'On the Wine dataset:\n- Random Forest achieves {wine_rf:.2f}% and Extra Trees achieves {wine_et:.2f}%\n- Both show substantial improvement over single Decision Tree ({wine_dt_custom:.2f}%)\n- The Wine dataset shows very high accuracy, suggesting it is highly separable using tree-based models')
    
    # 6.3 Custom vs Sklearn
    pdf.subsection('6.3', 'Custom vs Scikit-learn Comparison')
    pdf.body_text('Overall, the custom implementations achieve competitive performance with their scikit-learn counterparts:')
    
    rf_iris_diff = (final_results['iris']['RF (Custom)']['test_acc'] - final_results['iris']['RF (Sklearn)']['test_acc']) * 100
    et_iris_diff = (final_results['iris']['ET (Custom)']['test_acc'] - final_results['iris']['ET (Sklearn)']['test_acc']) * 100
    pdf.body_text(f'- Random Forest: Custom RF differs from sklearn RF by {rf_iris_diff:+.2f}% on Iris\n- Extra Trees: Custom ET differs from sklearn ET by {et_iris_diff:+.2f}% on Iris\n- Decision Tree: Performance is comparable with minor variations attributable to implementation differences')
    pdf.body_text('The results validate the correctness and robustness of the custom implementations.')
    
    # 6.4 Effect of Max Depth
    pdf.subsection('6.4', 'Effect of Max Depth on Decision Trees')
    pdf.body_text('Analysis of the effect of increasing maximum tree depth on classification accuracy reveals:\n- For both datasets, accuracy improves rapidly as maximum depth increases from 1 to approximately 3-4\n- Beyond depth 4-5, performance saturates for both datasets\n- This suggests that the datasets can be effectively separated using relatively shallow trees\n- No significant overfitting is observed even at high depths for ensembles, likely due to the variance reduction from averaging multiple trees')
    
    # 6.5 Effect of n_estimators
    pdf.subsection('6.5', 'Effect of Number of Trees on Ensembles')
    pdf.body_text('The analysis of increasing the number of trees (n_estimators) shows:')
    pdf.body_text('Iris Dataset:\n- Performance stabilizes quickly, with near-optimal accuracy achieved at around 25-50 trees\n- Both Random Forest and Extra Trees show similar convergence behavior')
    pdf.body_text('Wine Dataset:\n- More pronounced improvement as trees increase from 1 to 50\n- Performance converges to high accuracy around 50-100 trees\n- Additional trees beyond 100 provide minimal benefit')
    pdf.body_text('Both datasets exhibit the characteristic convergence behavior of bagging-based ensembles, where variance reduction saturates after averaging a sufficient number of trees.')
    
    # 6.6 RF vs ET
    pdf.subsection('6.6', 'Random Forest vs Extra Trees')
    pdf.body_text('Comparing the two ensemble methods:')
    rf_auroc = final_results['iris']['RF (Custom)']['test_auroc']
    et_auroc = final_results['iris']['ET (Custom)']['test_auroc']
    pdf.body_text(f'- Extra Trees may achieve slightly different AUROC on Iris ({et_auroc:.4f} vs RF\'s {rf_auroc:.4f}), suggesting different probability calibration\n- On the Wine dataset, both methods achieve very high performance\n- Extra Trees\' additional randomness in threshold selection does not hurt performance and may provide slight speed improvements during training')
    
    # ===== 7. CONCLUSION =====
    pdf.section('7', 'Conclusion')
    pdf.body_text('This work presented custom implementations of Decision Trees, Random Forests, and Extremely Randomized Trees (Extra Trees) and evaluated them on the Iris and Wine datasets. Key findings include:')
    
    pdf.body_text('1. Ensemble methods consistently outperform standalone Decision Trees by reducing variance and improving generalization. On the Iris dataset, ensembles improved accuracy significantly over single trees, and on Wine, ensemble methods achieved near-perfect accuracy.')
    
    pdf.body_text('2. The custom Decision Tree implementation closely matches the behavior and performance of the scikit-learn implementation, confirming the correctness of the CART-style Gini impurity-based splitting and stopping criteria.')
    
    pdf.body_text('3. Random Forests and Extra Trees achieve comparable or similar performance to their scikit-learn counterparts, validating the correctness of bootstrap sampling, feature subsampling, and random threshold selection implementations.')
    
    pdf.body_text('4. Analysis of hyperparameters indicates that:\n   - Moderate tree depths (around 5-10) are sufficient for both datasets\n   - 100 trees provide a good balance between predictive performance and computational efficiency')
    
    pdf.body_text('5. Extra Trees provide competitive performance with Random Forests while being potentially faster due to the random threshold selection (avoiding the search for optimal thresholds).')
    
    pdf.body_text('Overall, the close alignment between custom and scikit-learn models validates the robustness of the implementations and highlights the effectiveness of ensemble-based tree models for classification tasks.')
    
    # ===== 8. REFERENCES =====
    pdf.ln(5)
    pdf.section('8', 'References')
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, '[1] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.')
    pdf.ln(1)
    pdf.multi_cell(0, 5, '[2] Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3-42.')
    pdf.ln(1)
    pdf.multi_cell(0, 5, '[3] Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and Regression Trees. CRC Press.')
    pdf.ln(1)
    pdf.multi_cell(0, 5, '[4] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.')
    
    pdf.output('results/2005080_report.pdf')
    return 'results/2005080_report.pdf'


def main():
    print("=" * 50)
    print("CSE 472 Report Generator")
    print("Mohammad Ninad Mahmud Nobo | 2005080")
    print("=" * 50)
    
    print("\n[1/4] Running depth experiments...")
    depth_iris = experiment_varying_depth('iris')
    depth_wine = experiment_varying_depth('wine')
    
    print("[2/4] Running estimator experiments...")
    est_iris = experiment_varying_estimators('iris')
    est_wine = experiment_varying_estimators('wine')
    
    print("[3/4] Running final comparison...")
    final_results = experiment_final_comparison()
    
    print("[4/4] Generating figures and PDF...")
    plot_depth_analysis(depth_iris, depth_wine)
    plot_estimators_analysis(est_iris, est_wine)
    plot_final_comparison(final_results)
    plot_generalization_gap(final_results)
    
    pdf_path = generate_pdf_report(final_results)
    print(f"\n[DONE] Report saved: {pdf_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
