import os 
import logging
from datetime import datetime
from pathlib import Path

from config import LOG_DIR

# ----------------------------------------------------------------------------------------------------- #
# -------------------------------------------- Logging Setup ------------------------------------------ #
# ----------------------------------------------------------------------------------------------------- #
def setup_logging(app):
    """
    Sets up main logger and custom logger for the project
    Arguments:
        - app: the Flask app instance 
    Returns: 
        - logger: main project logger
    """
    os.makedirs(LOG_DIR, exist_ok=True) # ensure directory exists

    # create file for main log and query log
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    main_log_loc = Path(LOG_DIR, date_str+'_main.log')
    query_log_loc = Path(LOG_DIR, date_str+'_query.log')
    MAIN_LOG_FILE = str(main_log_loc)
    QUERY_LOG_FILE = str(query_log_loc)

    app.logger.setLevel(logging.INFO)   # level for app

    # set up the handler, formatting and level for main log
    main_file_handler = logging.FileHandler(MAIN_LOG_FILE)
    main_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [%(pathname)s:%(lineno)d]'
    ))
    main_file_handler.setLevel(logging.INFO)

    # define a custom level for formatted messages of query-response 
    CUSTOM = 25
    logging.addLevelName(CUSTOM, "Custom")
    def custom(self, message, *args, **kwargs):
        if self.isEnabledFor(CUSTOM):
            self._log(CUSTOM, message, args, **kwargs)
    logging.Logger.custom = custom

    # set up the handler, formatting, level and filter for custom log
    custom_file_handler = logging.FileHandler(QUERY_LOG_FILE)
    custom_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    custom_file_handler.setLevel(CUSTOM)
    custom_file_handler.addFilter(lambda record:record.levelno==CUSTOM)

    # --- Main project logger ---
    logger = logging.getLogger("ai-tool")
    logger.setLevel(logging.INFO)
    logger.addHandler(main_file_handler)
    logger.addHandler(custom_file_handler)

    # add main and custom handlers to app's logger
    app.logger.addHandler(main_file_handler)
    app.logger.addHandler(custom_file_handler)
    
    return logger

def get_logger(name=None):
    """
    Gets the main project logger when called in any project file
    Arguments: 
        - name: name of the file 
    Returns: 
        - the main project logger 
    """
    base = "ai-tool"
    return logging.getLogger(f"{base}.{name}" if name else base)
