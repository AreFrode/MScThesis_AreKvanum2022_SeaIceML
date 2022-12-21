import csv

from ast import literal_eval

import re

def read_config_from_csv(fpath):
    config = {}
    with open(fpath) as f:
        csv_reader = csv.reader(f, delimiter=',')
        rows = list(csv_reader)
        
    for key, value in zip(rows[0], rows[1]):
        try:
            config[key] = literal_eval(value)

        except (ValueError, SyntaxError):
            config[key] = value

    return config

def read_ice_edge_from_csv(fpath):
    ice_edge = {}
    
    with open(fpath) as f:
        csv_reader = csv.reader(f, delimiter=',')
        rows = list(csv_reader)

    for key, value in rows[1:]:
        ice_edge[key] = float(value) / 1000.  # ice_edge_length from verification metrics returns length in [m], want [km]

    return ice_edge

