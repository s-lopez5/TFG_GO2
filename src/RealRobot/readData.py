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



with open("training_data_1.pkl", "rb") as f:
    data = pickle.load(f)

input_data = data['inputs']  #Ya es np.array, listo para usar
output_data = data['outputs']

print(input_data)
print(output_data)
print(f"Nº de inputs: {input_data.shape}")
print(f"Nº de outputs: {output_data.shape}")
print(f"\nTotal de datos de entrada: {len(input_data)}\n")
print(f"\nTotal de datos de entrada: {len(output_data)}\n")