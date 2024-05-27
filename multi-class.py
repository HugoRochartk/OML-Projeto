#escolher classificador
from CLog_Dual_SemKernel import CLogD_MGB as module
import matplotlib.pyplot as plt

def save_intermed_csv(path, x, y):
    
    with open(path, "w") as f:
        for i in range(len(y)):
            f.write(','.join(map(str, x[i])) + f',{y[i]}\n')

    return path


def apply_w_classifiers(w_classifiers, x):

    results = {}

    for c in w_classifiers:
        results[c] = module.sigmoid(module.dot_product(w_classifiers[c], (1.0,) + x))
    
    return results


def plot_DB(x, y):
 
    colors = {1: 'r', 2: 'g', 3: 'b', 4: 'y'}

    coord_x = [p[0] for p in x]
    coord_y = [p[1] for p in x]
   
    for i in range(len(x)):
        plt.scatter(coord_x[i], coord_y[i], color=colors[y[i]], label=f'Classe {y[i]}')


    plt.legend()

  
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


database = "databases/multiclass1.csv"


x, y = module.take_data(database)
classes = set(y)
w_classifiers = {}

plot_DB(x, y)

for c in classes:
    x, old_y = module.take_data(database)
    y = [0.0 if elem != c else 1.0 for elem in old_y]
    c_database = save_intermed_csv(f"{database[:-4]}/class_{c}.csv", x, y)

    w, _, _ = module.apply_CLogD_MGB(c_database, 0.5, error_graph=False, accuracy=True, plot=False)
    w_classifiers[c] = w



new_input = (1.0, 0.5)
results = apply_w_classifiers(w_classifiers, new_input)
predicted_class = max(results, key=lambda k: results[k])


print('\nResults:', results)
print('Predicted class:', predicted_class)









