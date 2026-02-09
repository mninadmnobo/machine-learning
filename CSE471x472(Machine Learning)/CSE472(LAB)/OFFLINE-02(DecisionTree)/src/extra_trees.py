import numpy as np
from decision_tree import DecisionTreeClassifier


class ExtraTreesClassifier:
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.ensemble_trees = []
        np.random.seed(random_state)
    
    def fit(self, X, y):
        feature_data = np.array(X)
        target_labels = np.array(y)
        total_samples, total_features = feature_data.shape
        
        max_features_count = self.get_max_features_count(total_features)
        
        self.ensemble_trees = []
        
        for tree_index in range(self.n_estimators):
            bootstrap_indices = np.random.choice(total_samples, total_samples, replace=True)
            bootstrap_features = feature_data[bootstrap_indices]
            bootstrap_labels = target_labels[bootstrap_indices]
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                         min_samples_split=self.min_samples_split,
                                         random_state=self.random_state + tree_index)
            
            tree_modified = self.modify_tree_for_extra_trees(tree, max_features_count, total_features)
            tree_modified.fit(bootstrap_features, bootstrap_labels)
            
            self.ensemble_trees.append(tree_modified)
        
        return self
    
    def get_max_features_count(self, total_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(total_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return total_features
    
    def modify_tree_for_extra_trees(self, tree, max_features_count, total_features):
        original_find_best_split = tree.find_best_split
        
        def modified_find_best_split(feature_data, target_labels, total_features_param):
            selected_features = np.random.choice(total_features, max_features_count, replace=False)
            
            best_information_gain = -1
            best_split_feature = None
            best_split_threshold = None
            
            for feature_index in selected_features:
                feature_values = feature_data[:, feature_index]
                min_value = np.min(feature_values)
                max_value = np.max(feature_values)
                
                if min_value == max_value:
                    continue
                
                random_threshold = np.random.uniform(min_value, max_value)
                
                information_gain = tree.calculate_information_gain(target_labels, feature_values, random_threshold)
                
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_split_feature = feature_index
                    best_split_threshold = random_threshold
            
            return best_split_feature, best_split_threshold
        
        tree.find_best_split = modified_find_best_split
        return tree
    
    def predict(self, feature_data):
        feature_data = np.array(feature_data)
        total_samples = len(feature_data)
        
        predictions_from_all_trees = np.array([tree.predict(feature_data) 
                                               for tree in self.ensemble_trees])
        
        final_predictions = np.zeros(total_samples, dtype=int)
        
        for sample_index in range(total_samples):
            predictions_for_sample = predictions_from_all_trees[:, sample_index]
            final_predictions[sample_index] = self.get_majority_vote(predictions_for_sample)
        
        return final_predictions
    
    def get_majority_vote(self, predictions_array):
        unique_predictions, vote_counts = np.unique(predictions_array, return_counts=True)
        winning_class_index = np.argmax(vote_counts)
        return unique_predictions[winning_class_index]
    
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
