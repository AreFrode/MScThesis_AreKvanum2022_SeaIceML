import csv

from ast import literal_eval

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