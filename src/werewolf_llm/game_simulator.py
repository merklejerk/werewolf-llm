from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Set, Tuple
from werewolf_llm.roles import Role


@dataclass
class DiscussionTurn:
    """Represents a single statement made by a player during the discussion phase."""
    player: str
    statement: str

@dataclass
class PlayerInfo:
    """Holds all dynamic information about a player in the game."""
    initial_role: Role
    final_role: Role
    night_observation: str = ""


class GameSimulator:
    def __init__(self, players: List[str], roles: List[Role]):
        if len(set(players)) != len(players):
            raise ValueError("Player names must be unique.")
        if len(players) + 3 != len(roles):
            raise ValueError("Number of roles must be number of players + 3 (for center cards)")
        self.players: Set[str] = set(players)
        self.center_card_names: List[str] = [f"CenterCard{i+1}" for i in range(3)]
        self.roles: List[Role] = roles
        
        self.player_info: Dict[str, PlayerInfo] = {}
        self.center_roles: Dict[str, Role] = {}
        
        self.discussion_transcript: List[DiscussionTurn] = []
        self.votes: Dict[str, str] = {}

    def run_game(self) -> "GameSimulator":
        self._assign_initial_roles()
        self._simulate_night_phase()
        # Day phase and voting will be handled by the main game loop
        return self

    def _assign_initial_roles(self) -> None:
        shuffled_roles = self.roles.copy()
        random.shuffle(shuffled_roles)
        
        player_roles = {p: shuffled_roles.pop() for p in self.players}
        center_roles_list = shuffled_roles
        
        self.player_info = {
            p: PlayerInfo(initial_role=r, final_role=r) for p, r in player_roles.items()
        }
        self.center_roles = dict(zip(self.center_card_names, center_roles_list))

    def _simulate_night_phase(self) -> None:
        """
        Simulates the night phase by executing role actions in the correct order.
        """
        # The order of operations is critical in ONUW.
        night_actions = [
            self._werewolf_action,
            self._seer_action,
            self._robber_action,
            self._troublemaker_action,
        ]

        for action in night_actions:
            action()

    def _werewolf_action(self) -> None:
        werewolves = [p for p, info in self.player_info.items() if info.initial_role == Role.WEREWOLF]
        if len(werewolves) == 1:
            self.player_info[werewolves[0]].night_observation = "You are the lone werewolf."
        elif len(werewolves) > 1:
            for wolf in werewolves:
                other_wolves = [w for w in werewolves if w != wolf]
                self.player_info[wolf].night_observation = f"You see that {', '.join(other_wolves)} are also werewolves."

    def _seer_action(self) -> None:
        for player, info in self.player_info.items():
            if info.initial_role == Role.SEER:
                # Simple seer: looks at one player's card. A real implementation might see center cards.
                possible_targets = list(self.players - {player})
                if not possible_targets:
                    continue
                target_player = random.choice(possible_targets)
                target_role = self.player_info[target_player].initial_role
                self.player_info[player].night_observation = f"You observed {target_player} as a {target_role}."

    def _robber_action(self) -> None:
        for player, info in self.player_info.items():
            if info.initial_role == Role.ROBBER:
                possible_targets = list(self.players - {player})
                if not possible_targets:
                    continue
                target_player = random.choice(possible_targets)
                
                # Robber swaps their card with the target's card and sees their new role.
                robber_info = self.player_info[player]
                target_info = self.player_info[target_player]

                robber_original_role = robber_info.final_role
                target_original_role = target_info.final_role
                
                robber_info.final_role = target_original_role
                target_info.final_role = robber_original_role
                
                robber_info.night_observation = f"You robbed {target_player} and became a {target_original_role}."

    def _troublemaker_action(self) -> None:
        for player, info in self.player_info.items():
            if info.initial_role == Role.TROUBLEMAKER:
                possible_targets = list(self.players - {player})
                if len(possible_targets) < 2:
                    continue
                p1_name, p2_name = random.sample(possible_targets, 2)
                
                # Troublemaker swaps the cards of two other players.
                p1_info = self.player_info[p1_name]
                p2_info = self.player_info[p2_name]

                p1_original_role = p1_info.final_role
                p2_original_role = p2_info.final_role
                
                p1_info.final_role = p2_original_role
                p2_info.final_role = p1_original_role
                
                self.player_info[player].night_observation = f"You swapped the roles of {p1_name} and {p2_name}."


    def resolve_voting(self, votes: Dict[str, str]) -> Tuple[Optional[str], Dict[str, int]]:
        self.votes = votes
        vote_counts = {player: 0 for player in self.players}
        for voter, voted_for in votes.items():
            if voter not in self.players:
                raise ValueError(f"Invalid player name in votes (voter): {voter}")
            if voted_for not in self.players:
                raise ValueError(f"Invalid player name in votes (voted_for): {voted_for}")
            if voted_for in vote_counts:
                vote_counts[voted_for] += 1
        
        max_votes = 0
        executed_players = []
        for player, count in vote_counts.items():
            if count > max_votes:
                max_votes = count
                executed_players = [player]
            elif count == max_votes:
                executed_players.append(player)

        # In case of a tie, no one is executed in this simplified version
        if len(executed_players) == 1:
            return executed_players[0], vote_counts
        
        return None, vote_counts


    def determine_winner(self, executed_player: Optional[str]) -> Dict[str, float]:
        # Create a temporary combined dict of all final roles for easy lookup.
        all_final_roles = {p: info.final_role for p, info in self.player_info.items()}
        all_final_roles.update(self.center_roles)

        # 1. Check for Tanner win condition
        # The Tanner wins if they are executed, regardless of what else happens.
        if executed_player and all_final_roles[executed_player] == Role.TANNER:
            return {p: 1.0 if all_final_roles[p] == Role.TANNER else -1.0 for p in self.players}

        # 2. Determine if the Werewolf team wins
        werewolf_team_wins = True
        werewolves_in_game = any(r == Role.WEREWOLF for r in all_final_roles.values())
        
        if werewolves_in_game:
            # If there are werewolves, they win unless one of them is executed.
            if executed_player and all_final_roles[executed_player] == Role.WEREWOLF:
                 werewolf_team_wins = False # A werewolf was caught, so Village team wins.
            # If a villager is executed or no one is executed, Werewolf team wins.
        else:
            # If there are no werewolves, the Village team wins by default (as long as a Tanner wasn't executed).
            werewolf_team_wins = False

        # 3. Assign rewards based on the outcome
        rewards = {}
        for player in self.players:
            player_final_role = all_final_roles[player]
            is_on_werewolf_team = player_final_role == Role.WEREWOLF
            
            if is_on_werewolf_team:
                rewards[player] = 1.0 if werewolf_team_wins else -1.0
            else: # Villager, Seer, Robber, Troublemaker, etc.
                rewards[player] = -1.0 if werewolf_team_wins else 1.0
        
        return rewards

    # Accessors for the agent
    def get_player_final_role(self, player_name: str) -> Role:
        if player_name not in self.players:
            raise ValueError(f"Invalid player name: {player_name}")
        return self.player_info[player_name].final_role

    def get_player_initial_role(self, player_name: str) -> Role:
        if player_name not in self.players:
            raise ValueError(f"Invalid player name: {player_name}")
        return self.player_info[player_name].initial_role

    def get_player_night_observation(self, player_name: str) -> str:
        if player_name not in self.players:
            raise ValueError(f"Invalid player name: {player_name}")
        return self.player_info[player_name].night_observation
    
    def get_all_players(self) -> List[str]:
        return sorted(list(self.players))
    
    def get_all_roles_in_play(self) -> List[Role]:
        return self.roles
    
    def get_center_card_names(self) -> List[str]:
        return self.center_card_names
    
    def add_to_discussion_transcript(self, player: str, statement: str):
        if player not in self.players:
            raise ValueError(f"Invalid player name: {player}")
        self.discussion_transcript.append(DiscussionTurn(player=player, statement=statement))
    
    def add_discussion_turn(self, player: str, statement: str):
        """Add a discussion turn (alias for add_to_discussion_transcript)."""
        self.add_to_discussion_transcript(player, statement)
    
    def add_vote(self, player: str, voted_for: str):
        """Add a vote from a player."""
        if player not in self.players:
            raise ValueError(f"Invalid player name: {player}")
        self.votes[player] = voted_for
