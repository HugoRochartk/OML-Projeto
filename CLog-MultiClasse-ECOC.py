#escolher classificador
#from CLog_Dual_SemKernel import CLogD_MGB as module
from CLog_Dual_ComKernel import CLogDKPd_MGB as module

import matplotlib.pyplot as plt
from pprint import pprint



def plot_DB(x, y):
 
    colors = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'black'}

    coord_x = [p[0] for p in x]
    coord_y = [p[1] for p in x]
   
    for i in range(len(x)):
        plt.scatter(coord_x[i], coord_y[i], color=colors[y[i]], label=f'Classe {int(y[i])}')


    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

  
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


def save_intermed_csv(path, x, y):
    
    with open(path, "w") as f:
        for i in range(len(y)):
            f.write(','.join(map(str, x[i])) + f',{y[i]}\n')

    return path



def transpose(matrix):
    return [list(col) for col in zip(*matrix)]


def generate_codes_database(num_classes):

    if num_classes == 5:

        table = [[1] * 15,
                [0]*8 + [1]*7,
                [0]*4 + [1]*4 + [0]*4 + [1]*3,
                [0]*2 + [1]*2 + [0]*2 + [1]*2 + [0]*2 + [1]*2 + [0]*2 + [1],
                [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
                ]
                
    elif num_classes == 4:

        table = [[1] * 7,
                [0]*4 + [1]*3,
                [0]*2 + [1]*2 + [0]*2 + [1],
                [0,1,0,1,0,1,0]
                ]

    else:
        pass


    return table, transpose(table)



def apply_w_classifiers(input, w_classifiers):

    results = []

    for c in w_classifiers:
        float_predict = module.sigmoid(module.dot_product(w_classifiers[c], (1.0,) + input))
        if float_predict > 0.5:
            results.append(1)
        else:
            results.append(0)
    
    return results


def hamming_distance(l1, l2):
    
    if len(l1) == len(l2):
        d = 0
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                d += 1
        return d
    else:
        raise ValueError(f"{l1} and {l2} does not have the same lenght.")


def predict(input, table, w_classifiers):

    results = apply_w_classifiers(input, w_classifiers)

    distances = []

    print('\nResults: ', results)
    for i, line in enumerate(table):
        d = hamming_distance(results, line)
        distances.append((i+1, d))
    
    return (min(distances, key=lambda t: t[1]))[0]
    

def ECOC(database):

    x, old_y = module.take_data(database)
    table, classifiers = generate_codes_database(len(set(old_y)))
    path = f'{database[:-4]}-ECOC/'
    w_classifiers = {}

    plot_DB(x, old_y)

    for classifier_number, c in enumerate(classifiers):
        get_classes_at_1 = [i+1 for i, elem in enumerate(c) if elem == 1]
        y = [1.0 if int(elem) in get_classes_at_1 else 0.0 for elem in old_y]

        c_database = save_intermed_csv(path+f"classifier_{c}.csv", x, y)
        
        #w, _, _ = module.apply_CLogD_MGB(c_database, 0.5, error_graph=False, accuracy=True, plot=False, display_confusion_matrix=False)
        w, _, _ = module.apply_CLogDKPd_MGB(c_database, 0.5, 3, error_graph=False, accuracy=True, plot=False, display_confusion_matrix=False)

        w_classifiers[classifier_number] = w

    return table, w_classifiers
        
    

#train_path = "databases/multiclass1_train.csv"
#test_path = "databases/multiclass1_test.csv"

train_path = "databases/multiclass2_train.csv"
test_path = "databases/multiclass2_test.csv"


table, w_classifiers = ECOC(train_path)



''' 
ISTO É SO PARA 1 PONTO
FAZER UM FOR PARA TODOS OS PONTOS DA BD DE TESTE 
E CALCULAR ACCURACY

new_input = (0, 0)
predicted_class = predict(new_input, table, w_classifiers)

print('\nTable:')
pprint(table)

print(f'\nPrediction of {new_input}: Class {predicted_class}.')





x, y = module.take_data(test_path)

for input in x:
    ...


'''
final_model_accuracy = 0

x, y = module.take_data(test_path)
N = len(y)

for input, true_class in zip(x, y):
    new_input = input
    predicted_class = predict(new_input, table, w_classifiers)

    #print('\nTable:')
    #pprint(table)
    if predicted_class == true_class:
        final_model_accuracy += 1

    print(f'\nPrediction of {new_input}: Class {predicted_class}; True_class {int(true_class)}.')

print(f'\nFinal Model Accuracy: {final_model_accuracy/N}')