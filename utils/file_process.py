import os
import json
import pickle
import numpy as np


def ensure_dir(path):
    """
    建立对应的目录
    """
    if not os.path.exists(path):
        os.makedirs(path)


def format_filename(file_dir, filename_template, **kwargs):
    """
    格式化文件名
    """
    filename = os.path.join(file_dir, filename_template.format(**kwargs))
    return filename


def pickle_dump(filename, obj):
    """
    持久化对象
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

    print('Logging Info - Saved:', filename)


def pickle_load(filename):
    """
    从文件中获取对象
    """
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        print('Logging Info - Loaded:', filename)
    except EOFError:
        print('Logging Error - Cannot load:', filename)
        obj = None

    return obj

