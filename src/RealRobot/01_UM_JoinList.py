import numpy as np
import pickle

def leer_y_combinar_pickles(archivos):
    """
    Lee múltiples archivos pickle y combina sus datos.
    
    Args:
        archivos: Lista de rutas a los archivos pickle
        
    Returns:
        arrays_combinados: Lista con todos los arrays de 3 elementos
        utilidades_combinadas: Lista con todos los valores de utilidad
    """
    arrays_combinados = []
    utilidades_combinadas = []
    
    for archivo in archivos:
        print(f"Leyendo archivo: {archivo}")
        
        try:
            with open(archivo, 'rb') as f:
                datos = pickle.load(f)
            
            # Verificar si es el archivo UM_utility_data_80.pkl
            if "UM_utility_data_80" in archivo:
                print(f"  - Tratamiento especial para {archivo}")
                
                # Extraer datos de forma diferente
                arrays = datos['arrays']
                utilidades = datos['utility']
                
                # Agregar a las listas combinadas
                for array, utilidad in zip(arrays, utilidades):
                    arrays_combinados.append(array)
                    utilidades_combinadas.append(utilidad)
                
                print(f"  - Leídos {len(arrays)} pares de datos")
                
            else:
                # Tratamiento normal para otros archivos
                datos_objetivo = datos['datos_objetivo']
                
                # Separar arrays y utilidades
                for array, utilidad in datos_objetivo:
                    arrays_combinados.append(array)
                    utilidades_combinadas.append(utilidad)
                
                print(f"  - Leídos {len(datos_objetivo)} pares de datos")
            
            """
            with open(archivo, 'rb') as f:
                datos = pickle.load(f)
            
            # Extraer datos_objetivo
            datos_objetivo = datos['datos_objetivo']
            
            # Separar arrays y utilidades
            for array, utilidad in datos_objetivo:
                arrays_combinados.append(array)
                utilidades_combinadas.append(utilidad)
            
            print(f"  - Leídos {len(datos_objetivo)} pares de datos")
            """
        except FileNotFoundError:
            print(f"  - ERROR: Archivo no encontrado")
        except Exception as e:
            print(f"  - ERROR: {e}")
    
    return arrays_combinados, utilidades_combinadas

if __name__ == "__main__":

    print("="*80)
    print("UNIENDO ARCHIVOS PICKLE")
    print("="*80)

    #Nombres de los archivos pickle a unir
    archivos_pickle = [
        "UM_utility_data_80.pkl",
        "utility_data_with_models_1.pkl",
        "utility_data_with_models_2.pkl",
        "utility_data_with_models_3.pkl",
        "utility_data_with_models_4.pkl",   
        "utility_data_with_models_5.pkl",
        "utility_data_with_models_6.pkl",
        "utility_data_with_models_7.pkl",
        "utility_data_with_models_8.pkl",
        "utility_data_with_models_9.pkl",
        "utility_data_with_models_10.pkl",
        "utility_data_with_models_11.pkl",
        "utility_data_with_models_12.pkl",
        "utility_data_with_models_13.pkl",
        "utility_data_with_models_14.pkl",
        "utility_data_with_models_15.pkl",
        "utility_data_with_models_16.pkl",
        "utility_data_with_models_17.pkl",
        "utility_data_with_models_18.pkl",
        "utility_data_with_models_19.pkl",
        "utility_data_with_models_20.pkl"
    ]
    
    # Leer y combinar
    arrays, utilidades = leer_y_combinar_pickles(archivos_pickle)
    
    # Mostrar resultados
    print(f"\n{'='*50}")
    print(f"Total de datos combinados: {len(arrays)}")
    print(f"{'='*50}\n")
    
    # Mostrar algunos ejemplos
    print("Primeros 5 ejemplos:")
    for i in range(min(5, len(arrays))):
        print(f"  Array {i+1}: {arrays[i]}, Utilidad: {utilidades[i]}")
    
    #Convertir a arrays de numpy
    arrays_np = np.array(arrays)
    utilidades_np = np.array(utilidades)
    
    print(f"\nDimensiones del array combinado: {arrays_np.shape}")
    print(f"Rango de utilidades: [{utilidades_np.min()}, {utilidades_np.max()}]")
    
    #Guardar datos combinados
    datos_combinados = {
        'arrays': arrays_np,
        'utility': utilidades_np
    }
    
    with open("UM_utility_data_100.pkl", 'wb') as f:
        pickle.dump(datos_combinados, f)
    print("\nDatos combinados guardados en 'UM_utility_data_100.pkl'")
    