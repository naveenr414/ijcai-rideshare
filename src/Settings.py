# Configurable settings for each model 
settings_list = {}

def read_from_file(file_name):
    global settings_list
    
    f = open(file_name).read().split("\n")
    for line in f:
        if line!='':
            name = line.split(":")[0]
            value = eval(line.split(": ")[1])
            settings_list[name] = value

def has_value(name):
    return name in settings_list

def get_value(name):
    return settings_list[name]

def set_value(name,value):
    global settings_list

    settings_list[name] = value 
        