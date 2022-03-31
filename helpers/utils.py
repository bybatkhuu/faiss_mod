# -*- coding: utf-8 -*-

from pydantic import validate_arguments

from el_logging import logger

from ..helpers import validator


@validate_arguments
def clean_obj_dict(obj_dict: dict, cls_name: str):
    """Clean class name from object.__dict__ for str(object).

    Args:
        obj_dict (dict, required): Object dictionary by object.__dict__.
        cls_name (str , required): Class name by cls.__name__.

    Returns:
        dict: Clean object dictionary.
    """

    try:
        if validator.is_empty(obj_dict):
            raise ValueError("'obj_dict' argument value is empty!")

        if validator.is_empty(cls_name):
            raise ValueError("'cls_name' argument value is empty!")
    except ValueError as err:
        logger.error(err)
        raise

    _self_dict = obj_dict.copy()
    for _key in _self_dict.keys():
        _class_prefix = f"_{cls_name}__"
        if _key.startswith(_class_prefix):
            _new_key = _key.replace(_class_prefix, '')
            _self_dict[_new_key] = _self_dict.pop(_key)
    return _self_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def obj_to_repr(obj: object):
    """Modifying object default repr() to custom info.

    Args:
        obj (object, required): Any python object.

    Returns:
        str: String for repr() method.
    """

    try:
        if validator.is_empty(obj):
            raise ValueError("'obj' argument value is empty!")
    except ValueError as err:
        logger.error(err)
        raise

    return f"<{obj.__class__.__module__}.{obj.__class__.__name__} object at {hex(id(obj))}: " + "{" + f"{str(dir(obj)).replace('[', '').replace(']', '')}" + "}>"
