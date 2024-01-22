# Use the ``importlib.metadata`` package (known as ``importlib_metadata`` in Python < 3.8) to access the metadata of the package and return its version and author.

try:
    import importlib.metadata as importlib_metadata # For Python >= 3.8
except ImportError:
    import importlib_metadata as importlib_metadata # For Python < 3.8

try:
    # Use the in-built __package__ or __name__ variables to get metadata. Using __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib_metadata.metadata(__package__ or __name__)['version']
    __author__  = importlib_metadata.metadata(__package__ or __name__)['Author-email']
except importlib_metadata.PackageNotFoundError:
    __version__ = "Not installed as package yet!"
    __author__  = "Philip Cherian"
