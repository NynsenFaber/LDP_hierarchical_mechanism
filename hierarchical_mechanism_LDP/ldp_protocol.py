from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer
from pure_ldp.frequency_oracles.direct_encoding import DEClient, DEServer
from pure_ldp.frequency_oracles.hadamard_response import HadamardResponseClient, HadamardResponseServer
from pure_ldp.frequency_oracles.hadamard_mechanism import HadamardMechServer, HadamardMechClient
from pure_ldp.frequency_oracles.unary_encoding import UEClient, UEServer


def get_client_server(protocol: str, eps: float, depth: int, b: int) -> tuple:
    """
    Return the client and server for a given protocol

    GitHub: https://github.com/Samuel-Maddock/pure-LDP

    :param protocol: the protocol to use
    :param eps: privacy parameter
    :param depth: depth of the tree
    :param b: branching factor of the tree

    :return: the client and server
    """
    clients = []
    servers = []
    # create the clients and servers for each level of the tree, not for the root
    for level in range(1, depth):
        D = int(b ** level)
        # ------------- Local Hashing
        if protocol == 'local_hashing':
            clients.append(LHClient(epsilon=eps, d=D, use_olh=True))
            servers.append(LHServer(epsilon=eps, d=D, use_olh=True))

        # ------------- Direct Encoding
        elif protocol == 'direct_encoding':
            clients.append(DEClient(epsilon=eps, d=D))
            servers.append(DEServer(epsilon=eps, d=D))

        # ------------- Hadamard Response
        elif protocol == 'hadamard_response':
            server = HadamardResponseServer(epsilon=eps, d=D)
            servers.append(server)
            clients.append(HadamardResponseClient(epsilon=eps, d=D, hash_funcs=server.get_hash_funcs()))

        elif protocol == 'hadamard_mech':
            server = HadamardMechServer(epsilon=eps, d=D, use_optimal_t=True, t=None)
            servers.append(server)
            clients.append(HadamardMechClient(epsilon=eps, d=D, use_optimal_t=True, t=None))

        # ------------- Unary Encoding
        elif protocol == 'unary_encoding':
            clients.append(UEClient(epsilon=eps, d=D, use_oue=True))
            servers.append(UEServer(epsilon=eps, d=D, use_oue=True))

        else:
            raise ValueError(
                f"Protocol {protocol} not recognized, try 'local_hashing', 'direct_encoding' or 'hadamard_response'"
            )
    return clients, servers
