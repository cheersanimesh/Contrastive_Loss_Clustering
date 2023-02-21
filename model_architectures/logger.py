
def logger_single_write(file_path, mode, message):
    logger= open(file_path,mode)
    logger.write(message)
    logger.close()

def logger_multi_write(file_path, mode, messages):
    logger= open(file_path,mode)
    for msg in messages:
        logger.write(msg)
    logger.close()