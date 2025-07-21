from abc import ABC, abstractmethod
from typing import Any, Dict
from werewolf_llm.game_simulator import GameSimulator
from werewolf_llm.roles import Role
from pydantic import BaseModel, Field

class TurnOutput(BaseModel):
    """Represents the structured output of an agent for a single turn."""
    public_statement: str = Field(alias="publicStatement")
    private_role_guesses: Dict[str, str] = Field(alias="privateRoleGuesses")
    vote: str

    class Config:
        populate_by_name = True  # Allow both field names and aliases
        
    def to_json(self) -> str:
        """Serializes the object to a JSON string with camelCase field names."""
        return self.model_dump_json(by_alias=True, indent=2)

class Agent(ABC):
    def __init__(self, player_name: str):
        self.player_name = player_name

    @abstractmethod
    async def generate_turn_output(self, game: GameSimulator) -> TurnOutput:
        """
        Generates a complete, structured output for the agent's turn.
        This single method will produce the statement, guesses, and vote.
        """
        raise NotImplementedError
