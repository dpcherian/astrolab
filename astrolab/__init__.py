try:
    from importlib.metadata import metadata as __metadata__
    __version__ = __metadata__(__name__)['version']
    __author__  = __metadata__(__name__)['Author-email']
except:
    __version__ = "Not yet installed!"
    __author__  = "Philip Cherian"
