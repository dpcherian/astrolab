from importlib.metadata import metadata as __metadata__

__version__ = __metadata__("astrolab")['version']
__author__  = __metadata__("astrolab")['Author-email']
