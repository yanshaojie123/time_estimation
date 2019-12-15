import json
import os

# 加载配置文件中的变量
absPath = os.path.join(os.path.dirname(__file__), "config.json")
with open(absPath) as file:
    config = json.load(file)


def get_attribute(nametuple, defaultValue=None):
    try:
        if isinstance(nametuple, tuple):
            result = config[nametuple[0]]
            for i in range(1, len(nametuple)):
                result = result[nametuple[i]]
            return result
        elif isinstance(nametuple, str):
            return config[nametuple]
    except KeyError:
        return defaultValue
