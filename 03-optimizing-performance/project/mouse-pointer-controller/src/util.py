import logging

CPU_EXTENSION = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'


def check_layers_supported(engine, network, device):
    """
    Check if all layers of the network are supported
    :param engine: IECore
    :param network: IENetwork
    :param device: Inference device
    :return: True if all supported, False otherwise
    """
    layers_supported = engine.query_network(network, device_name=device)
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False
            logging.warning(f'Layer {l} is not supported on {device}')

    return all_supported
