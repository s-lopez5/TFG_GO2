import pickle
import numpy as np

with open("utility_data_with_models_9.pkl", "rb") as f:
    data = pickle.load(f)

"""
input_data = data['inputs']  #Ya es np.array, listo para usar
output_data = data['outputs']
"""
input_data = data['arrays']  #Ya es np.array, listo para usar
output_data = data['utility']


print(input_data)
print(output_data)
print(f"Nº de inputs: {input_data.shape}")
print(f"Nº de outputs: {output_data.shape}")
print(f"\nTotal de datos de entrada: {len(input_data)}\n")
print(f"\nTotal de datos de entrada: {len(output_data)}\n")