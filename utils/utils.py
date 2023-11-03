import logging
import os
import time

import colorlog
import torch

from utils.parameters import Params

def record_time(params: Params, t=None, name=None):
    if t and name and params.save_timing == name or params.save_timing is True:
        torch.cuda.synchronize()
        params.timing_data[name].append(round(1000 * (time.perf_counter() - t)))

def create_table(params: dict):
    data = f"\n| {'name' + ' ' * 21} | value    | \n|{'-'*27}|----------|"

    for key, value in params.items():
        # data += '\n' + f"| {(25 - len(key)) * ' ' }{key} | {value} |"
        data += f"\n| {key: <25} | {value} "
        
    return data

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
                  'INFO': 'bold_green',
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
