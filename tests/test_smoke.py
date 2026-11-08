import importlib

import pytest

SUBPACKAGES = [
    "pokerbots",
    "pokerbots.core",
    "pokerbots.sim",
    "pokerbots.tournament",
    "pokerbots.opponents",
    "pokerbots.bots",
    "pokerbots.bots.crisp",
    "pokerbots.bots.fuzzy",
    "pokerbots.bots.ml",
    "pokerbots.bots.simple",
    "pokerbots.analysis",
    "pokerbots.web",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(name)
