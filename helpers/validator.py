# -*- coding: utf-8 -*-

import numpy as np
from pydantic import validate_arguments

from el_logging import logger


@validate_arguments
def is_empty(val, trim: bool=False):
    """Empty checker method for any value.
    Works for None, val == '', len(val) == 0, and val.size == 0.

    Args:
        val  (any , required)        : Value to check.
        trim (bool, required) [False]: If 'val' is string, trim white spaces by strip() function.

    Returns:
        bool: True when empty, False for not empty.
    """

    if val is None:
        return True

    if isinstance(val, str):
        if trim:
            val = val.strip()

        if val == '':
            return True
    elif isinstance(val, list) or isinstance(val, dict) or isinstance(val, tuple) or isinstance(val, range) or isinstance(val, set):
        if len(val) == 0:
            return True
    elif isinstance(val, np.ndarray):
        if val.size == 0:
            return True
    return False


# @validate_arguments
# def _assert_attr_not_empty(obj, attr_name: str):
#     assert not _is_attr_empty(obj, attr_name), f"'{attr_name}' argument is empty!"
#     return True

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def is_attr_empty(obj: object, attr_name: str):
    """Empty checker method for object attribute.

    Args:
        obj       (object, required): Any object.
        attr_name (str   , required): Object's attibute name to check.

    Returns:
        bool: True when attribute is empty, False for not empty.
    """

    try:
        if is_empty(obj):
            raise ValueError("'obj' argument value is empty!")

        attr_name = attr_name.strip()
        if is_empty(attr_name):
            raise ValueError("'attr_name' argument value is empty!")
    except ValueError as err:
        logger.error(err)
        raise

    try:
        _val = getattr(obj, attr_name)
    except AttributeError:
        return True

    if is_empty(_val):
        return True
    return False
