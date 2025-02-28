from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer
from pure_ldp.frequency_oracles.direct_encoding import DEClient, DEServer
from pure_ldp.frequency_oracles.hadamard_response import HadamardResponseClient, HadamardResponseServer
from pure_ldp.frequency_oracles.unary_encoding import UEClient, UEServer

from .data_structure import TreeBary

import numpy as np
from typing import Union


def get_client_server(protocol: str, eps: float, depth: int, b: int) -> tuple:
    """
    Return the client and server for a given protocol

    GitHub: https://github.com/Samuel-Maddock/pure-LDP

    :param protocol: the protocol to use
    :param eps: privacy parameter
    :param D: the number of subintervals

    :return: the client and server
    """
    clients = []
    servers = []
    # create the clients and servers for each level of the tree, not for the root
    for level in range(1, depth):
        D = int(b ** level)
        if D > 10_000:
            # apply hadamard response for large D
            server = HadamardResponseServer(epsilon=eps, d=D)
            servers.append(server)
            clients.append(HadamardResponseClient(epsilon=eps, d=D, hash_funcs=server.get_hash_funcs()))
        else:
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

            # ------------- Unary Encoding
            elif protocol == 'unary_encoding':
                clients.append(UEClient(epsilon=eps, d=D, use_oue=True))
                servers.append(UEServer(epsilon=eps, d=D, use_oue=True))

            else:
                raise ValueError(
                    f"Protocol {protocol} not recognized, try 'local_hashing', 'direct_encoding' or 'hadamard_response'"
                )
    return clients, servers


def ldp_protocol(data: list[Union[int, float]],
                 eps: float,
                 tree: TreeBary,  # initial empty tree
                 protocol: str) -> list[LHServer]:
    """
    LDP protocol functions for the b-ary mechanism. It returns a list of servers with the privatized data for the
    b-adic decomposition of the domain (in intervals).

    Ref: Graham Cormode, Samuel Maddock, and Carsten Maple.
         Frequency Estimation under Local Differential Privacy. PVLDB, 14(11): 2046 - 2058, 2021

    GitHub: https://github.com/Samuel-Maddock/pure-LDP

    :param data: a list of data (already permuted possibly)
    :param eps: privacy parameter
    :param tree: the tree structure
    :param protocol: the protocol to use

    :return:
    """

    # this counter is used to keep track of the number of users that updated the tree at each level
    counts = np.zeros(depth, dtype=int)
    # iterate over the data and privatize it
    for i in range(len(data)):
        # sample a user
        user_value = data[i]
        # select a random level of the tree
        level = np.random.randint(1, depth)
        # select the index of the subinterval where the user belongs
        interval_index = tree.find_interval_index(user_value, level)
        # get the client and server (have index with an offset of 1)
        client = clients[level - 1]
        # privatize the data and send to the server
        priv_data = client.privatise(interval_index)
        servers[level - 1].aggregate(priv_data)
        counts[level - 1] += 1

    return servers, counts
