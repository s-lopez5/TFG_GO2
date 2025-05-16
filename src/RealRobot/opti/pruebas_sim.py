import threading
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Importa la clase NatNetClient (asegúrate de tener el módulo en tu PYTHONPATH o en el mismo directorio)
from optitrack import NatNetClient

# Variable global para almacenar los datos recibidos.
# Cada entrada será una tupla: (timestamp, id_cuerpo, x, y, z)
data_storage = []

def receive_rigid_body_frame(frame):
    """
    Callback que procesa los datos recibidos del servidor (Motive).
    Se asume que el objeto 'frame' es un diccionario con al menos:
      - 'timestamp': marca temporal
      - 'rigid_bodies': lista de diccionarios con los datos de cada cuerpo,
           donde cada cuerpo tiene 'id' y 'position' (una tupla (x, y, z)).
    """
    global data_storage, current_positions

    # Se utiliza el timestamp interno o se genera uno con time.time()
    timestamp = frame.get('timestamp', time.time())

    # Itera sobre cada cuerpo rígido del frame
    for rb in frame.get('rigid_bodies', []):
        rb_id = rb.get('id')
        pos = rb.get('position', (0, 0, 0))
        # Actualiza la posición actual para visualización
        current_positions[rb_id] = pos
        # Guarda los datos (timestamp, id, x, y, z)
        data_storage.append((timestamp, rb_id, pos[0], pos[1], pos[2]))

def natnet_thread():
    """
    Función que instancia y ejecuta el cliente NatNet.
    Se ejecuta en un hilo aparte para no bloquear la interfaz gráfica.
    """
    # Dirección del cliente y del servidor (modifícalas según tu red)
    client_address = "0.0.0.0"            # Escucha en todas las interfaces locales
    server_address = "192.168.1.100"        # IP del Intel NUC con Motive

    # Instancia el cliente y configura las direcciones
    client = NatNetClient()
    client.set_client_address(client_address)
    client.set_server_address(server_address)
    client.set_callback(receive_rigid_body_frame)
    
    # Inicia el proceso de escucha
    client.run()  # Se asume que este método es bloqueante

def animate(frame_index):
    """
    Función para actualizar la visualización en cada intervalo.
    Se borran los datos anteriores y se grafican las posiciones actuales (tomando X e Y).
    """
    plt.cla()  # Limpia la gráfica
    xs, ys, labels = [], [], []
    
    # Se itera sobre las posiciones actuales para graficarlas
    for rb_id, pos in current_positions.items():
        xs.append(pos[0])
        ys.append(pos[1])
        labels.append(str(rb_id))
    
    # Se crea un scatter plot
    plt.scatter(xs, ys, c='blue', s=50)
    
    # Se anotan los puntos con su ID
    for j, txt in enumerate(labels):
        plt.annotate(txt, (xs[j], ys[j]), textcoords="offset points", xytext=(5,5))
    
    plt.title('Posición en tiempo real de los cuerpos')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-5, 5)   # Ajusta estos límites a tu rango de captura
    plt.ylim(-5, 5)   # Ajusta estos límites a tu rango de captura

def on_close(event):
    """
    Función que se llama al cerrar la ventana. Se encarga de guardar los datos almacenados en formato CSV.
    """
    print("Guardando datos en 'data.csv'...")
    try:
        with open('data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Escribe la cabecera
            writer.writerow(['timestamp', 'id_cuerpo', 'x', 'y', 'z'])
            # Escribe cada entrada almacenada
            writer.writerows(data_storage)
        print("Datos guardados exitosamente.")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")

if __name__ == '__main__':
    # Inicia el cliente NatNet en un hilo separado para que no bloquee la GUI.
    client_thread = threading.Thread(target=natnet_thread, daemon=True)
    client_thread.start()

    # Configura la figura de Matplotlib para visualizar en tiempo real.
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, animate, interval=100)  # Actualizar cada 100 milisegundos

    # Conecta el evento de cierre de la ventana para guardar los datos.
    fig.canvas.mpl_connect('close_event', on_close)

    # Muestra la ventana gráfica (este llamado es bloqueante hasta que se cierre la ventana)
    plt.show()
