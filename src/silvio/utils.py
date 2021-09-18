"""
Contains various utility methods.
"""

from typing import Any, TypeVar, List, Callable, Optional

T = TypeVar('T')



def coalesce ( *arg ) -> Any :
    """
    Return the first non-null argument given. Emulates a nullish coalesce operator (PEP 505).
    """
    return next( (a for a in arg if a is not None), None )



def alldef ( *arg ) -> bool :
    """
    Returns true if all arguments are non-None.
    """
    return all([ a is not None for a in arg ])



def first ( array:List[T], criteria:Callable[[T],bool] ) -> Optional[T] :
    """
    Return the first item that validates the criteria.
    """
    return next( (item for item in array if criteria(item)), None )