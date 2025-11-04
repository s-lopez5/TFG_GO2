import sys
import time
import os

# Añadir el path del SDK si es necesario
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "NatNet_SDK", "samples", "PythonClient"))
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

from NatNet_SDK.samples.PythonClient.NatNetClient import NatNetClient
from NatNet_SDK.samples.PythonClient.NatNetClient import NatNetClient
from NatNet_SDK.samples.PythonClient.DataDescriptions import DataDescriptions
from NatNet_SDK.samples.PythonClient.MoCapData import MoCapData

def receive_new_frame_with_data(data_dict):
    order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", #type: ignore  # noqa F841
                  "rigidBodyCount", "skeletonCount", "labeledMarkerCount",
                  "timecode", "timecodeSub", "timestamp", "isRecording",
                  "trackedModelsChanged", "offset", "mocap_data"]
    dump_args = True
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += str(data_dict[key]) + " "
            out_string += "/"
        #print(out_string)


if __name__ == "__main__":
    
    optionsDict = {}
    optionsDict["clientAddress"] = "192.168.123.149"
    optionsDict["serverAddress"] = "192.168.123.164"
    optionsDict["use_multicast"] = False
    optionsDict["stream_type"] = 'd'
    stream_type_arg = None

    # This will create a new NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Streaming client configuration.
    streaming_client.new_frame_with_data_listener = receive_new_frame_with_data

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run(optionsDict["stream_type"])
    print("\n")
    print(streaming_client.get_print_level())
    print("\n")

    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")
    
