VERSION_INFO = {
    'major': 0,
    'minor': 1,
    'patch': 0,
    'dev': None
}

__version__ = f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"
if VERSION_INFO['dev'] is not None:
    __version__ += f"-dev{VERSION_INFO['dev']}"

def get_version():
    return __version__
