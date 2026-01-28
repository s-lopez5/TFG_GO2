import numpy as np
import pickle

# Nombres de los archivos
input_file = "training_data_merged.pkl"
output_file = "training_data_shuffled.pkl"

print("="*80)
print("MEZCLANDO ARRAYS")
print("="*80)

# Cargar el archivo
print(f"\nCargando {input_file}...")
with open(input_file, "rb") as f:
    data = pickle.load(f)
    inputs = data['inputs']
    outputs = data['outputs']

print(f"Forma de inputs: {inputs.shape}")
print(f"Forma de outputs: {outputs.shape}")
print(f"Total de muestras: {inputs.shape[0]}")

# Corregir forma si es necesario
if len(outputs.shape) == 3 and outputs.shape[1] == 1:
    outputs = outputs.squeeze(axis=1)
    print(f"Forma corregida de outputs: {outputs.shape}")

# Crear índices aleatorios para mezclar
print("\nMezclando el orden de las muestras...")
np.random.seed(42)  # Semilla para reproducibilidad (puedes cambiarla o eliminarla)
indices = np.arange(inputs.shape[0])
np.random.shuffle(indices)

# Aplicar la mezcla a ambos arrays
inputs_shuffled = inputs[indices]
outputs_shuffled = outputs[indices]

# Verificar que las formas se mantienen
print(f"\nForma después de mezclar:")
print(f"Inputs: {inputs_shuffled.shape}")
print(f"Outputs: {outputs_shuffled.shape}")

# Mostrar algunos índices para verificar la mezcla
print(f"\nPrimeros 10 índices después de mezclar: {indices[:10]}")

# Guardar el archivo mezclado
print(f"\nGuardando archivo mezclado como '{output_file}'...")
with open(output_file, "wb") as f:
    pickle.dump({
        'inputs': inputs_shuffled,
        'outputs': outputs_shuffled
    }, f)

print(f"\n✓ Archivo guardado exitosamente!")
print("="*80)
print(f"Resumen:")
print(f"  - Muestras totales: {inputs_shuffled.shape[0]}")
print(f"  - Orden mezclado aleatoriamente")
print(f"  - Guardado en: {output_file}")
print("="*80)