import os


def listdir_replays_only(path):
    return [
        dirent for dirent in os.listdir(path)
        if dirent.endswith('.roa')
        and os.path.isfile(os.path.join(path, dirent))
        ]


def version_to_dname(string):
    '''Convert x.x.x to xx_xx_xx'''
    return '_'.join([
        char if len(char) > 1 else '0' + char
        for char in string.split('.') if char.isdigit()
        ])


def dname_to_version(string):
    '''Convert xx_xx_xx to x.x.x'''
    return '.'.join([
        str(int(char))
        for char in string.split('_') if char.isdigit()
        ])

def ensure_directory_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)