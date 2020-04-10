from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG


def get_mylogger(filename=None):
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    formatter = Formatter("%(asctime)s %(levelname)8s %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if filename:
        file_handler = FileHandler(filename=filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
