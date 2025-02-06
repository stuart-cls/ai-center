# Logging abstraction to allow test installations without devioc
try:
    import devioc.log.get_module_logger as get_module_logger
except ImportError:
    import logging
    get_module_logger = logging.getLogger

__ALL__ = ['get_module_logger']