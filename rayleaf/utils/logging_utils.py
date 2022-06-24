logging_file = None


def log(msg: str = "") -> None:
    global logging_file

    print(msg, file=logging_file)
    print(msg)


def shutdown_log() -> None:
    global logging_file
    
    if logging_file is not None:
        logging_file.close()
    logging_file = None
