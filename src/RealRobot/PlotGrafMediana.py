import pickle
import matplotlib.pyplot as plt
import numpy as np

def visualizar_datos_utilidad(archivo_pickle):
    """
    Lee un archivo pickle con lista de tuplas (array[Distancia, Angulo Robot, Angulo Objetivo], utilidad)
    y muestra una gráfica de Distancia vs Utilidad con la mediana de distancias por nivel de utilidad
    
    Args:
        archivo_pickle: ruta al archivo pickle
    """
    #Cargar datos del archivo pickle
    with open(archivo_pickle, 'rb') as f:
        datos = pickle.load(f)
    
    #Extraer distancias y utilidades
    distancias = []
    utilidades = datos['utility']
    
    arrays = datos['arrays']

    for array in arrays:
        distancia = array[0]
        distancias.append(distancia)
    
    #Calcular la mediana de distancia para cada valor único de utilidad
    utilidades_unicas = sorted(set(utilidades))
    medianas_distancia = []
    
    for util_unica in utilidades_unicas:
        #Encontrar todas las distancias para esta utilidad
        distancias_util = [distancias[i] for i, u in enumerate(utilidades) if u == util_unica]
        mediana = np.median(distancias_util)
        medianas_distancia.append(mediana)
    
    #Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.scatter(distancias, utilidades, alpha=0.6, edgecolors='k', linewidth=0.5, label='Datos')
    
    #Graficar la mediana en rojo con línea
    plt.plot(medianas_distancia, utilidades_unicas, color='red', linewidth=2, marker='o', 
             markersize=5, label='Mediana de distancia', zorder=5)
    
    plt.xlabel('Distancia al Objetivo', fontsize=12)
    plt.ylabel('Valor de Utilidad', fontsize=12)
    plt.title('Relación entre Distancia y Utilidad', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    plt.xlim(0, 2)
    plt.ylim(0, 1.1)
    
    #Añadir información adicional
    plt.text(0.02, 0.98, f'Total de puntos: {len(distancias)}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('Grafica_dist_uti_100_Mediana.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    archivo = "UM_utility_data_100.pkl"
    
    try:
        visualizar_datos_utilidad(archivo)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo}'")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")