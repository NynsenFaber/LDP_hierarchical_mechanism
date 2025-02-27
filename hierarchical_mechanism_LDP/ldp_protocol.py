from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer
from pure_ldp.frequency_oracles.direct_encoding import DEClient, DEServer
from pure_ldp.frequency_oracles.hadamard_response import HadamardResponseClient, HadamardResponseServer
from pure_ldp.frequency_oracles.unary_encoding import UEClient, UEServer

from .data_structure import TreeBary

import numpy as np
from typing import Union


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
    intervals = tree.intervals

    clients = []
    servers = []
    depth = tree.depth
    # this counter is used to keep track of the number of users that updated the tree at each level
    counts = np.zeros(depth, dtype=int)
    # create the clients and servers for each level of the tree, not for the root
    for level in range(1, depth):
        # ------------- Local Hashing
        if protocol == 'local_hashing':
            clients.append(LHClient(epsilon=eps, d=len(intervals[level]), use_olh=True))
            servers.append(LHServer(epsilon=eps, d=len(intervals[level]), use_olh=True))

        # ------------- Direct Encoding
        elif protocol == 'direct_encoding':
            clients.append(DEClient(epsilon=eps, d=len(intervals[level])))
            servers.append(DEServer(epsilon=eps, d=len(intervals[level])))

        # ------------- Hadamard Response
        elif protocol == 'hadamard_response':
            server = HadamardResponseServer(epsilon=eps, d=len(intervals[level]))
            client = HadamardResponseClient(epsilon=eps, d=len(intervals[level]), hash_funcs=server.get_hash_funcs())
            servers.append(server)
            clients.append(client)

        # ------------- Unary Encoding
        elif protocol == 'unary_encoding':
            clients.append(UEClient(epsilon=eps, d=len(intervals[level]), use_oue=True))
            servers.append(UEServer(epsilon=eps, d=len(intervals[level]), use_oue=True))

        else:
            raise ValueError(
                f"Protocol {protocol} not recognized, try 'local_hashing', 'direct_encoding' or 'hadamard_response'"
            )

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
