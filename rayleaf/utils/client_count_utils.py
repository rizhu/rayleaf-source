def online(clients: list) -> list:
    """We assume all users are always online."""
    return sorted(clients.keys())


def count_selected_clients(selected_clients: list, clients: dict) -> dict:
    counts = {}
    for client_num in selected_clients:
        client_type = clients[client_num]
        counts[client_type] = counts.get(client_type, 0) + 1

    return counts


def client_counts_string(counts: dict) -> str:
    client_types = list(counts.keys())
    client_types.sort(key=lambda ClientType: ClientType.__name__)

    counts_str = ""
    count_format = "{count} {client_type_name}"

    if len(client_types) == 1:
        counts_str += count_format.format(count=counts[client_types[0]], client_type_name=client_types[0].__name__)
        if counts[client_types[0]] != 1:
            counts_str += "s"
    elif len(client_types) == 2:
        counts_str += count_format.format(count=counts[client_types[0]], client_type_name=client_types[0].__name__)
        if counts[client_types[0]] != 1:
            counts_str += "s"

        counts_str += " and "

        counts_str += count_format.format(count=counts[client_types[1]], client_type_name=client_types[1].__name__)
        if counts[client_types[1]] != 1:
            counts_str += "s"
    else:
        for client_type in client_types[:-1]:
            counts_str += count_format.format(count=counts[client_type], client_type_name=client_type.__name__)
            if counts[client_type] != 1:
                counts_str += "s"
            counts_str += ", "
        
        counts_str += "and "
        counts_str += count_format.format(count=counts[client_types[-1]], client_type_name=client_types[-1].__name__)
        if counts[client_types[-1]] != 1:
            counts_str += "s"
    
    return counts_str
