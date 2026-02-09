import numpy as np
from collections import Counter


class TreeNode:
    def __init__(self, split_feature_index=None, split_threshold=None, 
                 left_subtree=None, right_subtree=None, class_prediction=None):
        self.split_feature_index = split_feature_index
        self.split_threshold = split_threshold
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree
        self.class_prediction = class_prediction
    
    def is_leaf_node(self):
        return self.class_prediction is not None


class DecisionTreeClassifier:
    
    def __init__(self, max_depth=10, min_samples_split=2, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root_node = None
        np.random.seed(random_state)
    
    def fit(self, X, y):
        feature_data = np.array(X)
        target_labels = np.array(y)
        self.root_node = self.build_tree_recursive(feature_data, target_labels, depth=0)
        return self
    
    def build_tree_recursive(self, feature_data, target_labels, depth):
        total_samples, total_features = feature_data.shape
        unique_class_count = len(np.unique(target_labels))
        
        if (depth >= self.max_depth or 
            unique_class_count == 1 or 
            total_samples < self.min_samples_split):
            most_common_class = self.get_most_common_class(target_labels)
            return TreeNode(class_prediction=most_common_class)
        
        best_split_feature, best_split_threshold = self.find_best_split(feature_data, target_labels, total_features)
        
        if best_split_feature is None:
            most_common_class = self.get_most_common_class(target_labels)
            return TreeNode(class_prediction=most_common_class)
        
        left_mask = feature_data[:, best_split_feature] <= best_split_threshold
        right_mask = feature_data[:, best_split_feature] > best_split_threshold
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            most_common_class = self.get_most_common_class(target_labels)
            return TreeNode(class_prediction=most_common_class)
        
        left_child = self.build_tree_recursive(feature_data[left_mask], target_labels[left_mask], depth + 1)
        right_child = self.build_tree_recursive(feature_data[right_mask], target_labels[right_mask], depth + 1)
        
        return TreeNode(split_feature_index=best_split_feature, 
                       split_threshold=best_split_threshold,
                       left_subtree=left_child, right_subtree=right_child)
    
    def find_best_split(self, feature_data, target_labels, total_features):
        best_information_gain = -1
        best_split_feature = None
        best_split_threshold = None
        
        for feature_index in range(total_features):
            feature_values = feature_data[:, feature_index]
            unique_thresholds = np.unique(feature_values)
            
            for threshold in unique_thresholds:
                information_gain = self.calculate_information_gain(target_labels, feature_values, threshold)
                
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_split_feature = feature_index
                    best_split_threshold = threshold
        
        return best_split_feature, best_split_threshold
    
    def calculate_information_gain(self, target_labels, feature_values, threshold):
        parent_gini = self.calculate_gini_impurity(target_labels)
        
        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        total_samples = len(target_labels)
        left_samples_count = np.sum(left_mask)
        right_samples_count = np.sum(right_mask)
        
        gini_left = self.calculate_gini_impurity(target_labels[left_mask])
        gini_right = self.calculate_gini_impurity(target_labels[right_mask])
        
        weighted_child_gini = (left_samples_count / total_samples) * gini_left + (right_samples_count / total_samples) * gini_right
        
        information_gain = parent_gini - weighted_child_gini
        return information_gain
    
    def calculate_gini_impurity(self, target_labels):
        class_counts = np.bincount(target_labels.astype(int))
        class_probabilities = class_counts / len(target_labels)
        gini_impurity = 1 - np.sum(class_probabilities ** 2)
        return gini_impurity
    
    def get_most_common_class(self, target_labels):
        if len(target_labels) == 0:
            return 0
        class_counts = np.bincount(target_labels.astype(int))
        most_common_class = np.argmax(class_counts)
        return most_common_class
    
    def predict(self, feature_data):
        feature_data = np.array(feature_data)
        predictions = np.array([self.traverse_tree_for_prediction(sample, self.root_node) 
                               for sample in feature_data])
        return predictions
    
    def traverse_tree_for_prediction(self, sample, node):
        if node.is_leaf_node():
            return node.class_prediction
        
        if sample[node.split_feature_index] <= node.split_threshold:
            return self.traverse_tree_for_prediction(sample, node.left_subtree)
        else:
            return self.traverse_tree_for_prediction(sample, node.right_subtree)
    
    def predict_probability(self, feature_data):
        feature_data = np.array(feature_data)
        predictions = self.predict(feature_data)
        
        unique_classes = np.unique(predictions)
        num_classes = len(unique_classes)
        num_samples = len(predictions)
        
        probability_matrix = np.zeros((num_samples, num_classes))
        
        for sample_index, prediction in enumerate(predictions):
            class_index = np.where(unique_classes == prediction)[0][0]
            probability_matrix[sample_index, class_index] = 1.0
        
        return probability_matrix
