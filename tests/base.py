# coding=utf-8
import pprint
from typing import Any, Collection, Generic, Mapping, Optional, Type, TypeVar

from kgm.utils.common import get_all_subclasses, kwargs_or_empty

B = TypeVar('B')


class GenericTest(Generic[B]):
    """A generic test case."""
    #: The class
    cls: Type[B]

    #: The constructor keyword arguments
    kwargs: Optional[Mapping[str, Any]] = None

    #: The instance
    instance: B

    def setUp(self) -> None:
        self.instance = self.cls(**kwargs_or_empty(kwargs=self.kwargs))


class TestTests(Generic[B]):
    """Checks whether all subclasses have unittests."""
    base_cls: Type[B]
    base_test_cls: Type[GenericTest[B]]
    skip_cls: Collection[Type[B]] = frozenset()

    def test_testing(self):
        classes = get_all_subclasses(base_class=self.base_cls)
        tests = get_all_subclasses(base_class=self.base_test_cls)
        tested_classes = set(t.cls for t in tests if hasattr(t, 'cls'))
        uncovered_classes = classes.difference(tested_classes).difference(self.skip_cls)
        if len(uncovered_classes) > 0:
            raise NotImplementedError(f'No tests for \n{pprint.pformat(uncovered_classes)}')
