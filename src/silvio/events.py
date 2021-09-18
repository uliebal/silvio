"""
Events that serve as communication for and between host modules.
"""

from abc import ABC



class Event (ABC) :
    """
    Events are changes to the Host that will be handled by the Modules.
    """
    pass


# TODO: Try to replace initialize event with init code only.
# @dataclass
# class InitializeEvent ( Event ) :
#     pass
