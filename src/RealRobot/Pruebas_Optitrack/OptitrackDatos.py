import sys
import time
import os
import threading

# Añadir el path del SDK si es necesario
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "NatNet_SDK", "samples", "PythonClient"))
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

from NatNet_SDK.samples.PythonClient.NatNetClient import NatNetClient

# IP del servidor Motive y del cliente
server_ip = "192.168.123.112"
client_ip = "192.168.123.149"

# Variable para guardar el paquete recibido
paquete_recibido = None
frame_event = threading.Event()

"""
def on_new_frame(data_dict):
    
    global paquete_recibido
    paquete_recibido = data_dict

    print("----- MoCap Frame recibido -----")

    labeled_markers = data_dict.get("LabeledMarkers", [])
    if not labeled_markers:
        print("No se encontraron LabeledMarkers.")
    else:
        for i, marker in enumerate(labeled_markers):
            pos = marker.get("position", [0, 0, 0])
            size = marker.get("size", 0.0)
            print(f"Marcador {i + 1}:")
            print(f"  Posición: {pos}")
            print(f"  Tamaño: {size}")
    print("-------------------------------")

    frame_event.set()
"""

def on_new_frame(data_dict):
    # Solo para saber que llegó un frame
    print(f"MoCap Frame: {data_dict.get('frame_number')}")

    # Indica que llegó un frame para sincronizar el hilo que espera
    frame_event.set()

def on_labeled_marker(marker_id, position, size):
    print(f"Marcador etiquetado ID={marker_id}, Posición={position}, Tamaño={size}")


def recibir_un_paquete(server_ip, client_ip, multicast=False):
    """Conecta, espera un paquete y desconecta"""
    global paquete_recibido
    paquete_recibido = None
    frame_event.clear()

    # Iniciar cliente NatNet
    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(server_ip)
    client.set_use_multicast(multicast)

    client.new_frame_listener = on_new_frame
    client.labeled_marker_listener = on_labeled_marker
    
    if not client.run("d"):
        print("ERROR: no se pudo iniciar el cliente.")
        return None

    print("Esperando paquete...")

    # Esperar a que se reciba un paquete (máximo 3 segundos)
    recibido = frame_event.wait(timeout=3)

    client.shutdown()

    if recibido:
        return paquete_recibido
    else:
        print("No se recibió ningún paquete.")
        return None

# Bucle principal
if __name__ == "__main__":
    print("Cliente OptiTrack (recepción bajo demanda)")
    print("Comandos:")
    print("  m -> recibir un paquete de datos")
    print("  q -> salir")

    while True:
        cmd = input(">> ").strip().lower()
        if cmd == "m":
            paquete = recibir_un_paquete(server_ip, client_ip)
        elif cmd == "q":
            break