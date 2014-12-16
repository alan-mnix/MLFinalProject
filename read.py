import json
from pprint import pprint

def read(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    m = data[0]
    data.pop(0)
    return m, data