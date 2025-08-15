import time
from new_natnet_client.Client import NatNetClient, NatNetParams


def handle_frame(frame_data):
    """
    Función para procesar e imprimir información básica de cada frame recibido.
    """
    print(f"Frame #: {frame_data.frame_number}")
    print(f"Timestamp: {frame_data.timestamp}")

    # Si se reciben cuerpos rígidos, se recorren e imprimen sus datos:
    if hasattr(frame_data, 'rigid_bodies') and frame_data.rigid_bodies:
        for rb in frame_data.rigid_bodies:
            print(f"  Rigid Body ID: {rb.id}")
            print(f"    Posición: {rb.position}")
            print(f"    Orientación: {rb.orientation}")
    
    print("=" * 40)

def main():
    # Configuración de la conexión: Reemplaza las IPs con las correspondientes a tu red
    params = NatNetParams(
        server_address="192.168.50.112",         # IP del Intel NUC con Motive 3.2
        local_ip_address="192.168.50.149",       # IP del PC Ubuntu
        multicast_address="239.255.42.99"  # Dirección multicast por defecto
    )
    
    print("Intentando conectar al servidor NatNet (Motive 3.2)...")
    
    try:
        # Conexión al servidor utilizando la sintaxis del context manager
        with NatNetClient(params) as client:
            if client is None:
                print("Error: No se pudo establecer la conexión con el servidor NatNet.")
                return
            print("Conexión establecida. Recibiendo datos...")
            print("Presiona CTRL+C para detener la recepción.\n")
            requested_time = time.time_ns()
            # Bucle que itera de forma indefinida sobre cada frame recibido
            for frame in client.MoCap():
                handle_frame(frame)
                
                print(frame.rigid_body_data.rigid_bodies)
                if input("Out? [N/y]").upper() == 'Y': break
                print(frame.rigid_body_data.rigid_bodies)
                requested_time = time.time_ns()
                # Pequeña pausa para evitar imprimir demasiada información a gran velocidad
                time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nRecepción interrumpida por el usuario. Saliendo...")
    except Exception as e:
        print("Se produjo un error durante la conexión o recepción de datos:", e)

if __name__ == '__main__':
    main()