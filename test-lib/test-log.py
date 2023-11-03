import logging
import colorlog
import os
from logging import FileHandler

def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(filename)s - Line:%(lineno)d  - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'bold_blue',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)

logger = create_logger()

# # Configure basic logging settings with the custom formatter
# logging.basicConfig(level=logging.DEBUG, 
#                     format='%(log_color)s %(asctime)s - %(filename)s - Line:%(lineno)d - %(name)s - %(levelname)-8s - %(message)s')
# # cformat = '%(log_color)s' + format_str
# #         colors = {'DEBUG': 'reset',
# #                   'INFO': 'reset',
# #                   'WARNING': 'bold_yellow',
# #                   'ERROR': 'bold_red',
# #                   'CRITICAL': 'bold_red'}
                            
# # Set the custom formatter
# logger = logging.getLogger()

# # Create a logger
# logger = logging.getLogger("my_logger")

# # Create an HTML formatter
# class HTMLFormatter(logging.Formatter):
#     def format(self, record):
#         level = record.levelname
#         message = record.getMessage()
#         formatted = f"<p>{level}: {message}</p>\n"
#         return formatted

# # Create a FileHandler and set the HTML formatter
# file_handler = FileHandler('logs.html', mode='a', encoding='utf-8')
# html_formatter = HTMLFormatter()
# file_handler.setFormatter(html_formatter)

# # Add the FileHandler to the logger
# logger.addHandler(file_handler)


# Log messages
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")


# write logger to html file
