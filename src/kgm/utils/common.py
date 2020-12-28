import inspect
import itertools
import logging
from enum import Enum
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Set, Type, TypeVar, Union

T = TypeVar('T')


def enum_values(enum_cls: Type[Enum]) -> List:
    return [v.value for v in enum_cls]


def value_to_enum(enum_cls: Type[Enum], value: T) -> Enum:
    pos = [v for v in enum_cls if v.value == value]
    if len(pos) != 1:
        raise AssertionError(f'Could not resolve {value} for enum {enum_cls}. Available are {list(v for v in enum_cls)}.')
    return pos[0]


def identity(x: T) -> T:
    """The identity method."""
    return x


def get_all_subclasses(base_class: Type[T]) -> Set[Type[T]]:
    """Get a collection of all (recursive) subclasses of a given base class."""
    return set(base_class.__subclasses__()).union(s for c in base_class.__subclasses__() for s in get_all_subclasses(c))


def get_subclass_by_name(
    base_class: Type[T],
    name: str,
    normalizer: Optional[Callable[[str], str]] = None,
    exclude: Optional[Union[Collection[Type[T]], Type[T]]] = None,
) -> Type[T]:
    """Get a subclass of a base-class by name.

    :param base_class:
        The base class.
    :param name:
        The name.
    :param normalizer:
        An optional name normalizer, e.g. str.lower
    :param exclude:
        An optional collection of subclasses to exclude.

    :return:
        The subclass with matching name.
    :raises ValueError:
        If no such subclass can be determined.
    """
    if normalizer is None:
        normalizer = identity
    if exclude is None:
        exclude = set()
    if isinstance(exclude, type):
        exclude = {exclude}
    norm_name = normalizer(name)
    for subclass in get_all_subclasses(base_class=base_class).difference(exclude):
        if normalizer(subclass.__name__) == norm_name:
            return subclass
    subclass_dict = {normalizer(c.__name__): c for c in get_all_subclasses(base_class=base_class)}
    raise ValueError(f'{base_class} does not have a subclass named {norm_name}. Subclasses: {subclass_dict}.')


def argparse_bool(x):
    return str(x).lower() in {'true', '1', 'yes'}


def kwargs_or_empty(kwargs: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if kwargs is None:
        kwargs = {}
    return kwargs


def generate_experiments(
    grid_params: Optional[Mapping[str, List[Any]]] = None,
    explicit: Optional[List[Mapping[str, Any]]] = None,
) -> List[Mapping[str, Any]]:
    """
    Generate experiments for a parameter grid.

    :param grid_params:
        The grid parameters in format key -> list of choices
    :param explicit:
        Explicit experiment to add in form list(key -> value)
    :return:
        One list per worker.
    """
    # Default values
    if grid_params is None:
        grid_params = {}
    if explicit is None:
        explicit = []

    # Create search list for grid
    params = sorted(grid_params.keys())
    value_ranges = []
    for k in params:
        v = grid_params[k]
        if not isinstance(v, list) or isinstance(v, tuple):
            v = [v]
        value_ranges.append(v)
    search_list = [dict(zip(params, values)) for values in itertools.product(*value_ranges)]
    logging.info(f'Created grid search with {len(search_list)} runs.')

    # Add explicit runs
    if len(explicit) > 0:
        search_list += explicit
        logging.info(f'Added {len(explicit)} explicit runs.')

    # Consistent order
    return sorted(search_list, key=lambda config: tuple(hash(p) for p in sorted(config.items())))


def reduce_kwargs_for_method(
    method,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Prepare keyword arguments for a method.

    Drops excess parameters with warning, and checks whether arguments are provided for all mandantory parameters.
    """
    # Ensure kwargs is a dictionary
    kwargs = kwargs_or_empty(kwargs=kwargs)

    # compare keys with argument names
    signature = inspect.signature(method)
    parameters = set(signature.parameters.keys())

    # Drop arguments which are unexpected
    to_drop = set(kwargs.keys()).difference(parameters)
    if len(to_drop) > 0:
        dropped = {k: kwargs[k] for k in to_drop}
        logging.warning(f'Dropping parameters: {dropped}')
    kwargs = {k: v for k, v in kwargs.items() if k not in to_drop}

    # Check whether all necessary parameters are provided
    missing = set()
    for name, parameter in signature.parameters.items():
        if (parameter.default is parameter.empty) and parameter.name not in kwargs.keys() and parameter.name != 'self' and parameter.kind != parameter.VAR_POSITIONAL and parameter.kind != parameter.VAR_KEYWORD:
            missing.add(parameter.name)

    # check whether missing parameters are provided via kwargs
    missing = missing.difference(kwargs.get('kwargs', dict()).keys())

    if len(missing) > 0:
        raise ValueError(f'Method {method.__name__} missing required parameters: {missing}')

    return kwargs


def to_dot(
    config: Dict[str, Any],
    prefix: Optional[str] = None,
    separator: str = '.',
    function_to_name: bool = True,
) -> Dict[str, Any]:
    """Convert nested dictionary to flat dictionary.

    :param config:
        The potentially nested dictionary.
    :param prefix:
        An optional prefix.
    :param separator:
        The separator used to flatten the dictionary.
    :param function_to_name:
        Whether to convert functions to a string representation.

    :return:
        A flat dictionary where nested keys are joined by a separator.
    """
    result = dict()
    for k, v in config.items():
        if prefix is not None:
            k = f'{prefix}{separator}{k}'
        if isinstance(v, dict):
            v = to_dot(config=v, prefix=k, separator=separator)
        elif hasattr(v, '__call__') and function_to_name:
            v = {k: v.__name__ if hasattr(v, '__name__') else str(v)}
        else:
            v = {k: v}
        result.update(v)
    return result


def from_dot(
    dictionary: Mapping[str, Any],
    prefix: Optional[str] = None,
    separator: str = '.',
) -> Dict[str, Any]:
    """Convert flat dictionary to a nested dictionary.

    :param dictionary:
        The flat dictionary.
    :param prefix:
        An optional prefix.
    :param separator:
        The separator used to flatten the dictionary.

    :return:
        A nested dictionary where flat keys are split by a separator.
    """
    result = {}
    for k, v in dictionary.items():
        if prefix is not None:
            if k.startswith(prefix):
                k = k[:len(prefix)]
            else:
                logging.warning(f'k={k} does not start with prefix={prefix}.')
        key_sequence = k.split(sep=separator)
        sub_result = result
        for key in key_sequence[:-1]:
            if key not in sub_result:
                sub_result[key] = dict()
            sub_result = sub_result[key]
        sub_result[key_sequence[-1]] = v
    return result
