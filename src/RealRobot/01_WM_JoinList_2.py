import numpy as np
import pickle

# Nombres de los archivos pickle a unir
file1 = "training_data_merged_685.pkl"
file2 = "training_data_7f.pkl"
file3 = "training_data_8f.pkl"
file4 = "training_data_9f.pkl"
file5 = "training_data_10.pkl"
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

#Cargar el tercer archivo
print(f"\nCargando {file3}...")
with open(file3, "rb") as f:
    data3 = pickle.load(f)
    inputs3 = data3['inputs']
    outputs3 = data3['outputs']

print(f"Forma original de inputs3: {inputs3.shape}")
print(f"Forma original de outputs3: {outputs3.shape}")

#Cargar el cuarto archivo
print(f"\nCargando {file4}...")
with open(file4, "rb") as f:
    data4 = pickle.load(f)
    inputs4 = data4['inputs']
    outputs4 = data4['outputs']

print(f"Forma original de inputs4: {inputs4.shape}")
print(f"Forma original de outputs4: {outputs4.shape}")

#Cargar el quinto archivo
print(f"\nCargando {file5}...")
with open(file5, "rb") as f:
    data5 = pickle.load(f)
    inputs5 = data5['inputs']
    outputs5 = data5['outputs']

print(f"Forma original de inputs5: {inputs5.shape}")
print(f"Forma original de outputs5: {outputs5.shape}")

#Unir los arrays
print("\nUniendo arrays...")
inputs_merged = np.concatenate([inputs1, inputs2, inputs3, inputs4, inputs5], axis=0)
outputs_merged = np.concatenate([outputs1, outputs2, outputs3, outputs4, outputs5], axis=0)

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
print(f"  - Archivo 3: {inputs3.shape[0]} muestras")
print(f"  - Archivo 4: {inputs4.shape[0]} muestras")
print(f"  - Archivo 5: {inputs5.shape[0]} muestras")
print(f"  - Total: {inputs_merged.shape[0]} muestras")
print(f"  - Guardado en: {output_file}")
print("="*80)