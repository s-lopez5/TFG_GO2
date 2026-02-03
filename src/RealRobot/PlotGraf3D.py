import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualizar_datos_utilidad(archivo_pickle):
    """
    Lee un archivo pickle con lista de tuplas (array[Distancia, Angulo Robot, Angulo Objetivo], utilidad)
    y muestra una gráfica 3D de Distancia vs Ángulo Objetivo vs Utilidad
    
    Args:
        archivo_pickle: ruta al archivo pickle
    """
    # Cargar datos del archivo pickle
    with open(archivo_pickle, 'rb') as f:
        datos = pickle.load(f)
    
    #Extraer distancias, ángulos al objetivo y utilidades
    distancias = []
    angulos_objetivo = []
    utilidades = datos['utility']
    
    arrays = datos['arrays']

    for array in arrays:
        distancia = array[0]
        angulo_objetivo = array[1]  #Segunda columna es el ángulo al objetivo
        distancias.append(distancia)
        angulos_objetivo.append(angulo_objetivo)
    
    #Crear la gráfica 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    #Crear el scatter plot 3D con colormap basado en la utilidad
    scatter = ax.scatter(distancias, utilidades, angulos_objetivo, 
                        c=utilidades, cmap='viridis', 
                        alpha=0.6, edgecolors='k', linewidth=0.5,
                        s=20)
    
    ax.set_xlabel('Distancia al Objetivo', fontsize=11, labelpad=10)
    ax.set_ylabel('Valor de Utilidad', fontsize=11, labelpad=10)
    ax.set_zlabel('Ángulo al Objetivo', fontsize=11, labelpad=10)
    ax.set_title('Relación entre Distancia, Ángulo al Objetivo y Utilidad', 
                 fontsize=14, fontweight='bold', pad=20)
    
    #Añadir barra de color
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Utilidad', fontsize=10)
    
    #Configurar límites
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.1)
    ax.set_zlim(-3.5, 3.5)
    
    #Añadir información adicional
    ax.text2D(0.02, 0.98, f'Total de puntos: {len(distancias)}', 
              transform=ax.transAxes, 
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    #Ajustar el ángulo de vista para mejor visualización
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('Grafica_3D_dist_angulo_uti_100_3D.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    archivo = "UM_utility_data_100.pkl"
    
    try:
        visualizar_datos_utilidad(archivo)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo}'")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")