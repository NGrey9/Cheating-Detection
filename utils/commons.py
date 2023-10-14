import yaml

def read_yaml(filepath):
    '''
    Arguments: filepath {str}: yaml filepath string
    Returns: data_loaded {dict}: a dictionary contains the contents of the yaml file
    '''
    with open(filepath,'r') as f:
        data_loaded = yaml.safe_load(f)
        return data_loaded