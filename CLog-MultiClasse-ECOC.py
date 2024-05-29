from CLog_Dual_SemKernel import CLogD_MGB as module
import matplotlib.pyplot as plt
import itertools
from pprint import pprint

def max_indices(arr):
    max_value = max(arr)
    
    return [index + 1 for index, value in enumerate(arr) if value == max_value]

def generate_codes_database(num_classes):
    combinations = list(itertools.product([0, 1], repeat=num_classes))
    combinations = combinations[len(combinations)//2:-1]
    
    new_labels = []
    for combination in combinations:
        new_y = []
        indexes = max_indices(combination)
        
        new_y = [ 1 if int(label) in indexes else 0 for i, label in enumerate(y)]
        new_labels.append((combination, new_y))
    
    return new_labels


def predict(new_point, classifiers):
    # falta terminar a prediction
    for classifier in classifiers:
        weights = classifier[1]

def ECOC(x, y):
    new_labels = generate_codes_database(len(set(y)))
    classifiers = []
    
    for label in new_labels:
        classifier = label[0]
        y = label[1]
        database = (x, y)
        # Plot = True está a dar erro 
        res = module.apply_CLogD_MGB(database, 0.5, error_graph=False, accuracy=True, plot=False, display_confusion_matrix=False, path=False)
        weights = res[0]
        
        classifiers.append((classifier, weights))
        
    return classifiers
        
    
    
        
        
database = "databases/Multiclass1.csv"
x, y = module.take_data(database)
classifiers = ECOC(x, y)
new_input = (0.5, 0.25)



