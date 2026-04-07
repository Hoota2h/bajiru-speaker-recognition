import logging

logger = logging.getLogger("baji")


def set_up_logging(terminal_level_name: str = "INFO", file_level_name: str = "NONE") -> None:
    terminal_level = logging.getLevelName(terminal_level_name)
    file_level = 100 if file_level_name == "NONE" else logging.getLevelName(file_level_name)
    logger.setLevel(min([terminal_level, file_level]))
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # create console handler with a higher log level
    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(terminal_level)
    # create formatter and add it to the handlers
    terminal_handler.setFormatter(formatter)
    logger.addHandler(terminal_handler)
    if file_level < logging.CRITICAL:
        file_handler = logging.FileHandler("baji_recognition.log")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
