from typing import Any, Dict
from werewolf_llm.game_simulator import GameSimulator
from werewolf_llm.roles import Role
import json
import random

class Agent:
    def __init__(self, player_name: str):
        self.player_name = player_name

    async def get_public_statement(self, game: GameSimulator) -> str:
        """
        Generates a public statement based on the agent's role and observations.
        This is a very basic, rule-based implementation.
        """
        my_final_role = game.get_player_final_role(self.player_name)
        observation = game.get_player_night_observation(self.player_name)

        if my_final_role == Role.SEER:
            return f"I am the Seer. {observation}"
        if my_final_role == Role.WEREWOLF:
            # Lie and claim to be a villager
            return "I am a Villager. I did nothing suspicious."
        if my_final_role == Role.ROBBER:
            return f"I started as the Robber. {observation}"
        if my_final_role == Role.TROUBLEMAKER:
            return f"I am the Troublemaker. {observation}"
        
        return "I am a Villager. I don't have much information."

    async def get_vote(self, game: GameSimulator) -> str:
        """
        Decides who to vote for based on the discussion.
        Basic strategy: vote for someone who is not yourself.
        """
        possible_targets = [p for p in game.get_all_players() if p != self.player_name]
        if not possible_targets:
            # This should not happen in a normal game, but as a fallback, vote for self.
            return self.player_name
        return random.choice(possible_targets)

    async def get_private_guesses(self, game: GameSimulator) -> Dict[str, str]:
        """
        Makes a private guess about everyone's final role.
        This is a naive implementation.
        """
        guesses = {}
        all_players = game.get_all_players()
        center_cards = game.get_center_card_names()
        all_entities = all_players + center_cards
        my_final_role = game.get_player_final_role(self.player_name)
        
        # Guess own role correctly
        guesses[self.player_name] = my_final_role.value

        # Naively guess others
        remaining_players = [p for p in all_entities if p != self.player_name]
        remaining_roles = game.get_all_roles_in_play().copy()
        remaining_roles.remove(my_final_role)
        random.shuffle(remaining_roles)

        for i, player in enumerate(remaining_players):
            if i < len(remaining_roles):
                guesses[player] = remaining_roles[i].value
            else:
                # Fallback if role list is exhausted
                guesses[player] = Role.VILLAGER.value
        
        return guesses


    async def generate_turn_output(self, game: GameSimulator, voting_phase: bool = False) -> str:
        public_statement = await self.get_public_statement(game)
        private_guesses = await self.get_private_guesses(game)

        output: Dict[str, Any] = {
            "PUBLIC_STATEMENT": public_statement,
            "PRIVATE_ROLE_GUESSES": private_guesses,
        }
        if voting_phase:
            output["VOTE"] = await self.get_vote(game)
        
        return json.dumps(output, indent=2)
