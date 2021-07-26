from pathlib import Path

def get_path(path_string):
    return Path(path_string).__str__()