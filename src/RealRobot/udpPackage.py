import socket

def receive_motive_unicast(ip="192.168.123.149", data_port=1511, buffer_size=4096):
    """
    Escucha paquetes UDP unicast enviados desde Motive (OptiTrack NatNet).
    """
    # Crear socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # En unicast no hace falta unirse a un grupo multicast
    sock.bind((ip, data_port))

    print(f"Escuchando paquetes UDP (unicast) de Motive en {ip}:{data_port}...\n(CTRL+C para detener)\n")

    try:
        while True:
            data, addr = sock.recvfrom(buffer_size)
            print(f"📦 Paquete recibido de {addr} | tamaño: {len(data)} bytes")
            # Muestra los primeros bytes en formato hexadecimal
            print("Bytes iniciales:", data[:32].hex(), "...\n")
    except KeyboardInterrupt:
        print("\nCerrando socket...")
    finally:
        sock.close()

if __name__ == "__main__":
    receive_motive_unicast()