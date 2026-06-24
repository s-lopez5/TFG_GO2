import numpy as np
import pickle

def modificar_utilidades(archivo_entrada, archivo_salida):
    """
    Lee un archivo pickle y modifica las utilidades invirtiendo la escala:
    - 0.1 ↔ 1.0
    - 0.2 ↔ 0.9
    - 0.3 ↔ 0.8
    - 0.4 ↔ 0.7
    - 0.5 ↔ 0.6
    
    Args:
        archivo_entrada: ruta del archivo pickle a leer
        archivo_salida: ruta donde guardar el resultado
    """
    #Leer el archivo pickle
    with open(archivo_entrada, 'rb') as f:
        datos = pickle.load(f)
    
    #Extraer arrays y utilidades
    arrays = datos['arrays']
    utilidades = datos['utility']
    
    print(f"Datos originales:")
    print(f"  - Número de elementos: {len(arrays)}")
    print(f"  - Utilidades únicas: {np.unique(utilidades)}")
    
    #Convertir a numpy array si no lo es
    utilidades = np.array(utilidades)

    #Invertir la escala usando la fórmula 1.1 - utilidad
    """
    - 0.1 → {1.1 - 0.1}
    - 0.5 → {1.1 - 0.5}
    - 1.0 → {1.1 - 1.0}
    """
    utilidades_modificadas = 1.1 - utilidades
    
    #Redondear para evitar problemas de precisión de punto flotante
    utilidades_modificadas = np.round(utilidades_modificadas, 1)
    
    #Crear el nuevo diccionario con datos modificados
    datos_modificados = {
        'arrays': arrays,
        'utility': utilidades_modificadas
    }
    
    print(f"\nDatos modificados:")
    print(f"  - Utilidades: {utilidades}")
    print(f"  - Utiliadades inversa: {utilidades_modificadas}")
    print(f"  - Utilidades únicas: {np.unique(utilidades_modificadas)}")
    
    #Guardar el archivo modificado
    with open(archivo_salida, 'wb') as f:
        pickle.dump(datos_modificados, f)
    
    print(f"\nArchivo guardado en: {archivo_salida}")
    
    return datos_modificados
    
# Ejemplo de uso
if __name__ == "__main__":
    modificar_utilidades('UM_utility_data_100.pkl', 'UM_utility_data_100_inverso.pkl')