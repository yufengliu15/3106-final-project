from pokerbots.bots.simple.always_all_in import AlwaysAllInBot
from pokerbots.bots.simple.random_bot import RandomBot
from pokerbots.bots.simple.scripted import (
    CallingMachineBot,
    LooseAggressiveBot,
    ScriptedBot,
    SwitcherBot,
    TightAggressiveBot,
    TightPassiveBot,
)

SCRIPTED_POOL = (TightAggressiveBot, TightPassiveBot, LooseAggressiveBot,
                 CallingMachineBot, AlwaysAllInBot, RandomBot, SwitcherBot)

__all__ = ["RandomBot", "AlwaysAllInBot", "CallingMachineBot", "LooseAggressiveBot",
           "ScriptedBot", "SwitcherBot", "TightAggressiveBot", "TightPassiveBot",
           "SCRIPTED_POOL"]
