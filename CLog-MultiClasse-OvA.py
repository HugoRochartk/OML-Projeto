#escolher classificador
#from CLog_Dual_SemKernel import CLogD_MGB as module
from CLog_Dual_ComKernel import CLogDKPd_MGB as module
#from CLog_Primal import CLog_MGB as module

import matplotlib.pyplot as plt
import pandas as pd
import time

def double_digit_sec(secs):
    if secs < 10:
        return f"0{secs}"
    return str(secs)


def save_intermed_csv(path, x, y):
    
    with open(path, "w") as f:
        for i in range(len(y)):
            f.write(','.join(map(str, x[i])) + f',{y[i]}\n')

    return path


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


def build_light_mnist(train, test):

    to_save = "databases/light_mnist_"

    for i, path in enumerate([train, test]):

        df = pd.read_csv(path)
        label_column = 'label'


        min_samples_per_class = df[label_column].value_counts().min()
        min_samples_per_class = min_samples_per_class // 100

    
        reduced_dfs = []

    
        for digit in range(10):

            digit_df = df[df[label_column] == digit]
            reduced_digit_df = digit_df.sample(n=min_samples_per_class, random_state=42)
            reduced_dfs.append(reduced_digit_df)

        
        reduced_df = pd.concat(reduced_dfs)
        reduced_df = reduced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        if i == 0:
            aux = 'train'
        else:
            aux = 'test'

        reduced_df.to_csv(to_save + aux + '.csv', index=False)




def mnist_organize(train, test):


    for path in [train, test]:

        df = pd.read_csv(path, skiprows=1, header=None)

        cols = df.columns.tolist()  
        fst_col = cols.pop(0)  
        cols.append(fst_col)  

        df = df[cols]

        
        df.to_csv(path, index=False, header=False)



def apply_w_classifiers(w_classifiers, x):

    results = {}

    for c in w_classifiers:
        results[c] = module.sigmoid(module.dot_product(w_classifiers[c], (1.0,) + x))
    
    return results


# ---------------------- MNIST SECTION ---------------
# Controlar tempo no MNIST
#start_time = time.time()

#train_path = "databases/mnist_train.csv"
#test_path = "databases/mnist_test.csv"

#mnist_organize(train_path, test_path)
#build_light_mnist(train_path, test_path)


#train_path = "databases/light_mnist_train.csv"
#test_path = "databases/light_mnist_test.csv"

# ------------------------------------------------------


train_path = "databases/multiclass2_train.csv"
test_path = "databases/multiclass2_test.csv"

#train_path = "databases/multiclass1_train.csv"
#test_path = "databases/multiclass1_test.csv"




x, old_y = module.take_data(train_path)
y = old_y
classes = set(map(int, y))
w_classifiers = {}

plot_DB(x, y)


for c in classes:
    
    y = [0.0 if elem != c else 1.0 for elem in old_y]

    c_database = save_intermed_csv(f"{train_path[:-4]}/class_{c}.csv", x, y)

    w, _, _ = module.apply_CLogDKPd_MGB(c_database, 0.5, 3, error_graph=False, accuracy=True, plot=True, display_confusion_matrix=False)
    #MULTICLASS1: #w, _, _ = module.apply_CLogD_MGB(c_database, 0.5, error_graph=False, accuracy=True, plot=False, display_confusion_matrix=False)

    #PARA O MNIST: w, _ = module.apply_CLog_MGE(c_database, (0,) * 785, 0.5, error_graph=False, accuracy=True, plot=False, display_confusion_matrix=False)

    w_classifiers[c] = w





''' 
------------------------------------- MNIST SECTION ----------------------------------

end_time = time.time()
time_dif = end_time - start_time

print("Tempo de treino: " + f"{int(time_dif//60)}min" + double_digit_sec(int(time_dif - ((time_dif//60)*60))) + "s")

--------------------------------------------------------------------------------------------
'''

final_model_accuracy = 0

x_test, y_test = module.take_data(test_path)
N = len(y_test)

print('-----------------------------------')
for i in range(N):
    results = apply_w_classifiers(w_classifiers, x_test[i])
    predicted_class = max(results, key=lambda k: results[k])
    if predicted_class == y_test[i]:
        final_model_accuracy += 1
    print(f'Results: {results}; Predicted class: {predicted_class}; True class: {y_test[i]}')

print(f'\nFinal Model Accuracy: {final_model_accuracy/N}')
    










