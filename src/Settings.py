import sys

# Configurable settings for each model 
settings_list = {}

def read_from_file(file_name):
    global settings_list
    
    f = open(file_name).read().split("\n")
    for line in f:
        if line!='':
            name = line.split(":")[0]
            if "," in line.split(": ")[1]:
                value = line.split(": ")[1].split(",")
            else:
                value = eval(line.split(": ")[1])
            settings_list[name] = value

    print("Finished reading settings")

def read_from_arguments():
    global settings_list
    
    f = sys.argv[1:]
    for i in range(0,len(f),2):
        name = f[i]
        val = f[i+1]
        if "," in val:
            val = val.split(",")
        else:
            val = eval(val)
        settings_list[name] = val
    print(settings_list)

def has_value(name):
    return name in settings_list

def get_value(name):
    return settings_list[name]

def set_value(name,value):
    global settings_list

    settings_list[name] = value 
        
