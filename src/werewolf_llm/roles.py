from enum import Enum


class Role(Enum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    SEER = "Seer"
    TROUBLEMAKER = "Troublemaker"
    ROBBER = "Robber"
    TANNER = "Tanner"

    def __str__(self) -> str:
        return self.value
