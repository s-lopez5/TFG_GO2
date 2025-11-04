import numpy as np
import pickle

# Nombres de los archivos pickle a unir
file1 = "training_data_1.pkl"
file2 = "training_data_2.pkl"
output_file = "training_data_merged.pkl"

print("="*80)
print("UNIENDO ARCHIVOS PICKLE")
print("="*80)

#Cargar el primer archivo
print(f"\nCargando {file1}...")
with open(file1, "rb") as f:
    data1 = pickle.load(f)
    inputs1 = data1['inputs']
    outputs1 = data1['outputs']

print(f"Forma original de inputs1: {inputs1.shape}")
print(f"Forma original de outputs1: {outputs1.shape}")

#Corregir forma si es necesario para outputs1
if len(outputs1.shape) == 3 and outputs1.shape[1] == 1:
    outputs1 = outputs1.squeeze(axis=1)
    print(f"Forma corregida de outputs1: {outputs1.shape}")

#Cargar el segundo archivo
print(f"\nCargando {file2}...")
with open(file2, "rb") as f:
    data2 = pickle.load(f)
    inputs2 = data2['inputs']
    outputs2 = data2['outputs']

print(f"Forma original de inputs2: {inputs2.shape}")
print(f"Forma original de outputs2: {outputs2.shape}")

#Corregir forma si es necesario para outputs2
if len(outputs2.shape) == 3 and outputs2.shape[1] == 1:
    outputs2 = outputs2.squeeze(axis=1)
    print(f"Forma corregida de outputs2: {outputs2.shape}")

#Unir los arrays
print("\nUniendo arrays...")
inputs_merged = np.concatenate([inputs1, inputs2], axis=0)
outputs_merged = np.concatenate([outputs1, outputs2], axis=0)

print(f"\nForma final de inputs: {inputs_merged.shape}")
print(f"Forma final de outputs: {outputs_merged.shape}")
print(f"Total de muestras: {inputs_merged.shape[0]}")

#Guardar el archivo unido
print(f"\nGuardando archivo unido como '{output_file}'...")
with open(output_file, "wb") as f:
    pickle.dump({
        'inputs': inputs_merged,
        'outputs': outputs_merged
    }, f)

print(f"\n✓ Archivo guardado exitosamente!")
print("="*80)
print(f"Resumen:")
print(f"  - Archivo 1: {inputs1.shape[0]} muestras")
print(f"  - Archivo 2: {inputs2.shape[0]} muestras")
print(f"  - Total: {inputs_merged.shape[0]} muestras")
print(f"  - Guardado en: {output_file}")
print("="*80)