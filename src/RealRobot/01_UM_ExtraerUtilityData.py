import numpy as np
import pickle

def read_WM_data(archivos):
    """
    Lee múltiples archivos pickle y combina sus datos.
    
    Args:
        archivos: Lista de rutas a los archivos pickle
        
    Returns:
        
    """

    utility_list = []

    for archivo in archivos_pickle:
        print(f"Leyendo archivo: {archivo}")
            
        try:
            with open(archivo, 'rb') as f:
                data = pickle.load(f)
                inputs = data['inputs']
                outputs = data['outputs']
            
            #Obtener los últimos 10 datos y asignar valores de 0.0 a 1.0
            ultimos = outputs[-10:] if len(outputs) >= 10 else outputs
            datos_aux = []

            for i, elemento in enumerate(reversed(ultimos)):
                # Asignar valores de 1.0, 0.9, 0.8, ..., 0.0
                valor = 1.0 - (i * 0.1)
                datos_aux.append((elemento, valor))
    
            # Invertir la lista para mantener el orden original
            datos_aux.reverse()

            utility_list.extend(datos_aux)  
            
        except FileNotFoundError:
            print(f"  - ERROR: Archivo no encontrado")
        except Exception as e:
            print(f"  - ERROR: {e}") 

    utility_list_np = np.array(utility_list, dtype=object) 

    print(f"Total de elementos: {len(utility_list)}")
    print(f"Forma del array: {utility_list_np.shape}")
    
    # Dividir en dos arrays separados
    arrays_np = np.array([elemento for elemento, _ in utility_list])
    utility_np = np.array([valor for _, valor in utility_list])
    
    print(f"Forma de arrays: {arrays_np.shape}")
    print(f"Forma de utility: {utility_np.shape}")
    
    return arrays_np, utility_np


if __name__ == "__main__":
    
    #Nombres de los archivos pickle a unir
    archivos_pickle = [
        "training_data_1f.pkl",
        "training_data_2f.pkl",
        "training_data_3f.pkl",
        "training_data_4f.pkl",
        "training_data_5f.pkl",
        "training_data_6f.pkl",
        "training_data_7f.pkl",
        "training_data_8f.pkl",
        "training_data_9f.pkl"
    ]

    #Devuelve la lista con los valores de utilidad
    arrays, utility = read_WM_data(archivos_pickle)
    
    print(f"\nDimensiones del array combinado: {arrays.shape}")
    print(f"Rango de utilidades: [{utility.min()}, {utility.max()}]")
    
    #Guardar datos combinados
    datos_combinados = {
        'arrays': arrays,
        'utility': utility
    }
    
    with open("UM_utility_data.pkl", 'wb') as f:
        pickle.dump(datos_combinados, f)
    print("\nDatos combinados guardados en 'UM_utility_data_merged_15.pkl'")
    