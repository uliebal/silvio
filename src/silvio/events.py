"""
Events that serve as communication for and between host modules.
"""

from abc import ABC
from typing import Callable


class Event (ABC) :
    """
    Events are changes to the Host that will be handled by the Modules.
    """
    pass


EventEmitter = Callable[[Event],None]

EventLogger = Callable[[str],None]
