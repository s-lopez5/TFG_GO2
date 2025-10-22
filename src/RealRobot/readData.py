import pickle
import numpy as np
"""
def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

train_data = load_data("lista_obs.pkl")
input_data, output_data = map(list, zip(*train_data))

X = np.array(input_data)

#print(f"\nTotal de datos de entrada: {len(X)}\n")
print(train_data.shape)
print(train_data)
"""

with open("training_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data['inputs']  # Ya es np.array, listo para usar
y_train = data['outputs']

print(X_train.shape)
print(X_train)
print(y_train.shape)
print(y_train)