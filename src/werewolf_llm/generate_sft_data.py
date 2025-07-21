import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import sys
import asyncio

from llama_cpp import Llama
from .game_simulator import GameSimulator, DiscussionTurn
from .roles import Role
from .agent import Agent, TurnOutput


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTDataPoint:
    """Represents a single training example for SFT."""
    prompt: str
    completion: str
    metadata: Dict[str, Any]


class RuleBasedAgent(Agent):
    """Enhanced rule-based agent for generating high-quality SFT data."""
    
    def __init__(self, player_name: str, strategy_variant: str = "standard"):
        super().__init__(player_name)
        self.strategy_variant = strategy_variant
        
    async def generate_turn_output(self, game: GameSimulator) -> TurnOutput:
        """Generates a complete, structured output for the agent's turn."""
        public_statement = await self._get_public_statement(game)
        private_guesses = await self._get_private_guesses(game)
        vote = await self._get_vote(game)

        return TurnOutput(
            publicStatement=public_statement,
            privateRoleGuesses=private_guesses,
            vote=vote,
        )

    async def _get_public_statement(self, game: GameSimulator) -> str:
        """Generate role-appropriate public statements with strategy variants."""
        my_initial_role = game.get_player_initial_role(self.player_name)
        observation = game.get_player_night_observation(self.player_name)
        
        # Add some randomness to statement patterns
        if my_initial_role == Role.VILLAGER:
            return self._villager_statement(game)
        elif my_initial_role == Role.WEREWOLF:
            return self._werewolf_statement(game)
        elif my_initial_role == Role.SEER:
            return self._seer_statement(game, observation)
        elif my_initial_role == Role.ROBBER:
            return self._robber_statement(game, observation)
        elif my_initial_role == Role.TROUBLEMAKER:
            return self._troublemaker_statement(game, observation)
        elif my_initial_role == Role.TANNER:
            return self._tanner_statement(game)
        else:
            return "I am a Villager. I don't have any special information."
    
    def _villager_statement(self, game: GameSimulator) -> str:
        statements = [
            "I am a Villager. I have no special abilities or information.",
            "I'm just a regular Villager. I didn't do anything during the night.",
            "I am a Villager and I'm looking for suspicious behavior from others.",
        ]
        return random.choice(statements)
    
    def _werewolf_statement(self, game: GameSimulator) -> str:
        # Werewolves typically lie about their role
        if self.strategy_variant == "aggressive":
            fake_claims = [
                "I am the Seer. I need to share what I observed.",
                "I am a Villager, and I think we need to be very careful about who we trust.",
            ]
        else:
            fake_claims = [
                "I am a Villager. I don't have any special information to share.",
                "I'm just a regular Villager trying to figure out who the Werewolf is.",
            ]
        return random.choice(fake_claims)
    
    def _seer_statement(self, game: GameSimulator, observation: str) -> str:
        if self.strategy_variant == "truthful":
            return f"I am the Seer. {observation}"
        else:
            # Sometimes Seers might be cautious about revealing information
            if random.random() < 0.8:  # 80% chance to be truthful
                return f"I am the Seer. {observation}"
            else:
                return "I have some information that might be helpful, but I want to hear from others first."
    
    def _robber_statement(self, game: GameSimulator, observation: str) -> str:
        if "became a" in observation:
            return f"I started as the Robber. {observation}"
        else:
            return "I am the Robber, but I didn't get to use my ability."
    
    def _troublemaker_statement(self, game: GameSimulator, observation: str) -> str:
        if self.strategy_variant == "secretive":
            return "I am the Troublemaker. I made some changes during the night."
        else:
            return f"I am the Troublemaker. {observation}"
    
    def _tanner_statement(self, game: GameSimulator) -> str:
        # Tanner wants to be executed, so might act suspicious
        statements = [
            "I am a Villager, but I've been acting strange because I'm nervous.",
            "I don't have much to contribute. Maybe we should just vote randomly?",
            "I'm a Villager, but honestly, I think you should all be suspicious of me.",
        ]
        return random.choice(statements)
    
    async def _get_vote(self, game: GameSimulator) -> str:
        """Strategic voting based on role and discussion."""
        my_initial_role = game.get_player_initial_role(self.player_name)
        all_players = game.get_all_players()
        other_players = [p for p in all_players if p != self.player_name]
        
        if not other_players:
            return self.player_name
        
        # Simple strategy: werewolves try to vote for non-werewolves
        if my_initial_role == Role.WEREWOLF:
            # Try to vote for someone who claimed Seer (high threat)
            for turn in game.discussion_transcript:
                if turn.player != self.player_name and "Seer" in turn.statement:
                    return turn.player
        
        # Tanner wants to be voted for, but can't vote for themselves in this simplified version
        if my_initial_role == Role.TANNER:
            # Vote somewhat randomly to avoid being too obvious
            pass
        
        # Default: vote for a random other player
        return random.choice(other_players)
    
    async def _get_private_guesses(self, game: GameSimulator) -> Dict[str, str]:
        """Generate strategic private role guesses."""
        my_initial_role = game.get_player_initial_role(self.player_name)
        observation = game.get_player_night_observation(self.player_name)
        
        all_players = game.get_all_players()
        center_cards = game.get_center_card_names()
        all_entities = all_players + center_cards
        
        guesses = {}
        
        # Determine what we think our current role is based on our initial role and observations
        my_current_role = my_initial_role
        if my_initial_role == Role.ROBBER and "became a" in observation:
            # Extract the new role from the observation
            parts = observation.split("became a ")[1]
            role_name = parts.strip().rstrip(".")
            try:
                my_current_role = Role(role_name)
            except ValueError:
                pass
        
        # Start with own role (what we think we currently are)
        guesses[self.player_name] = my_current_role.value
        
        # Use night observation to inform guesses
        observed_roles = self._extract_observed_roles(observation, all_entities)
        
        # Update guesses based on discussion transcript
        discussion_claims = self._analyze_discussion_claims(game.discussion_transcript)
        
        # Combine observations and claims to make educated guesses
        remaining_entities = [e for e in all_entities if e != self.player_name]
        remaining_roles = game.get_all_roles_in_play().copy()
        remaining_roles.remove(my_current_role)
        
        # Apply known information first
        for entity, role in observed_roles.items():
            if entity in remaining_entities:
                guesses[entity] = role.value
                remaining_entities.remove(entity)
                if role in remaining_roles:
                    remaining_roles.remove(role)
        
        # Apply discussion claims with some skepticism
        for entity, claimed_role in discussion_claims.items():
            if entity in remaining_entities and claimed_role in remaining_roles:
                # Believe claims with some probability (varies by our role)
                belief_probability = 0.7 if my_initial_role != Role.WEREWOLF else 0.3
                if random.random() < belief_probability:
                    guesses[entity] = claimed_role.value
                    remaining_entities.remove(entity)
                    remaining_roles.remove(claimed_role)
        
        # Fill in remaining guesses randomly
        random.shuffle(remaining_roles)
        for i, entity in enumerate(remaining_entities):
            if i < len(remaining_roles):
                guesses[entity] = remaining_roles[i].value
            else:
                guesses[entity] = Role.VILLAGER.value
        
        return guesses
    
    def _extract_observed_roles(self, observation: str, all_entities: List[str]) -> Dict[str, Role]:
        """Extract role information from night observations."""
        observed = {}
        
        if "observed" in observation and "as a" in observation:
            # Seer observation: "You observed PlayerB as a Werewolf"
            parts = observation.split("observed ")[1].split(" as a ")
            if len(parts) == 2:
                player_name = parts[0].strip()
                role_name = parts[1].strip().rstrip(".")
                try:
                    role = Role(role_name)
                    if player_name in all_entities:
                        observed[player_name] = role
                except ValueError:
                    pass
        
        elif "became a" in observation:
            # Robber observation: "You robbed PlayerX and became a Villager"
            parts = observation.split("became a ")[1]
            role_name = parts.strip().rstrip(".")
            try:
                role = Role(role_name)
                # The robber now has this role
                observed[self.player_name] = role
            except ValueError:
                pass
        
        return observed
    
    def _analyze_discussion_claims(self, transcript: List[DiscussionTurn]) -> Dict[str, Role]:
        """Extract role claims from discussion transcript."""
        claims = {}
        
        for turn in transcript:
            statement = turn.statement
            player = turn.player
            
            # Look for role claims
            for role in Role:
                role_patterns = [
                    f"I am the {role.value}",
                    f"I am a {role.value}",
                    f"I'm the {role.value}",
                    f"I'm a {role.value}",
                ]
                
                for pattern in role_patterns:
                    if pattern in statement:
                        claims[player] = role
                        break
        
        return claims


class SFTDataGenerator:
    """Main class for generating SFT training data."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.llm = None

        if model_path:
            logger.info(f"Loading LLM from {model_path}")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=8,
                verbose=False
            )
    
    def generate_game_configs(self, num_games: int) -> List[Tuple[List[str], List[Role]]]:
        """Generate diverse game configurations."""
        configs = []
        
        # Define some standard configurations
        base_configs = [
            # 3 players + 3 center = 6 roles total
            (["Alice", "Bob", "Charlie"], 
             [Role.VILLAGER, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.ROBBER, Role.TROUBLEMAKER]),
            
            # 4 players + 3 center = 7 roles total  
            (["Alice", "Bob", "Charlie", "Diana"],
             [Role.VILLAGER, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.ROBBER, Role.TROUBLEMAKER, Role.TANNER]),
            
            # 5 players + 3 center = 8 roles total
            (["Alice", "Bob", "Charlie", "Diana", "Eve"],
             [Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.ROBBER, Role.TROUBLEMAKER, Role.TANNER]),
            
            # 6 players + 3 center = 9 roles total
            (["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
             [Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.ROBBER, Role.TROUBLEMAKER, Role.TANNER]),
        ]
        
        for _ in range(num_games):
            config = random.choice(base_configs)
            configs.append(config)
        
        return configs
    
    async def simulate_game_for_sft(self, players: List[str], roles: List[Role]) -> List[SFTDataPoint]:
        """Simulate a single game and extract SFT training examples."""
        game = GameSimulator(players, roles)
        game.run_game()
        
        # Create rule-based agents with different strategy variants
        strategy_variants = ["standard", "aggressive", "truthful", "secretive"]
        agents = {}
        for player in players:
            variant = random.choice(strategy_variants)
            agents[player] = RuleBasedAgent(player, variant)
        
        data_points = []
        
        # Simulate discussion phase (1-2 rounds)
        num_rounds = random.randint(1, 2)
        
        for round_num in range(num_rounds):
            for player in players:
                # Generate prompt for this player's turn
                prompt = self._generate_prompt(game, player)
                
                # Generate response using rule-based agent
                agent = agents[player]
                turn_output = await agent.generate_turn_output(game)
                
                # Optionally rephrase the public statement using LLM
                if self.llm:
                    original_statement = turn_output.public_statement
                    rephrased_statement = await self._rephrase_statement(original_statement, player, game)
                    if rephrased_statement:
                        turn_output.public_statement = rephrased_statement
                
                completion_json = turn_output.to_json()

                # Create data point
                is_voting_phase = (round_num == num_rounds - 1)
                metadata = {
                    "player": player,
                    "round": round_num,
                    "initial_role": game.get_player_initial_role(player).value,
                    "final_role": game.get_player_final_role(player).value,
                    "night_observation": game.get_player_night_observation(player),
                    "is_final_round": is_voting_phase,
                    "strategy_variant": agent.strategy_variant,
                }
                
                data_point = SFTDataPoint(
                    prompt=prompt,
                    completion=completion_json,
                    metadata=metadata
                )
                data_points.append(data_point)
                
                # Add the statement to the game transcript for subsequent players
                game.add_to_discussion_transcript(player, turn_output.public_statement)
        
        return data_points
    
    def _generate_prompt(self, game: GameSimulator, player: str) -> str:
        """Generate the input prompt for a player's turn."""
        
        # System instruction
        system_instruction = (
            "You are playing One Night Ultimate Werewolf. Your objective is to secure victory for your team. "
            "Formulate strategic statements and votes. Respond with a JSON object containing publicStatement, privateRoleGuesses, and vote fields."
        )
        
        # Game state
        all_players = game.get_all_players()
        possible_roles = [role.value for role in game.get_all_roles_in_play()]
        center_cards = game.get_center_card_names()
        
        game_state = f"""Players: {', '.join(all_players)}
Possible Roles in Play: {', '.join(possible_roles)}
Center Cards: {', '.join(center_cards)} (hidden from players)"""
        
        # Player's role and observations
        initial_role = game.get_player_initial_role(player)
        night_observation = game.get_player_night_observation(player)
        
        player_id_info = player
        role_info = f"You are the {initial_role.value}."
        observation_info = f"{night_observation}" if night_observation else "You have no special night observations."
        
        # Discussion transcript
        transcript_lines = []
        for turn in game.discussion_transcript:
            transcript_lines.append(f'{turn.player}: "{turn.statement}"')
        
        transcript_section = "\n".join(transcript_lines) if transcript_lines else "(No statements yet)"
        
        # Combine all parts
        prompt = f"""[SYSTEM_INSTRUCTION] {system_instruction}

[GAME_STATE]
{game_state}

[YOUR_PLAYER_ID] {player_id_info}

[YOUR_INITIAL_ROLE] {role_info}
[YOUR_NIGHT_OBSERVATIONS] {observation_info}

[DISCUSSION_TRANSCRIPT]
{transcript_section}"""
        
        return prompt
    
    async def _rephrase_statement(self, original_statement: str, player: str, game: GameSimulator) -> Optional[str]:
        """Use LLM to rephrase statements for linguistic diversity."""
        if not self.llm:
            return None
        
        final_role = game.get_player_final_role(player)
        
        rephrase_prompt = f"""You are helping to create training data for a Werewolf game AI. 
Please rephrase the following statement to make it more natural and varied while preserving the exact same meaning and strategic intent.

Original role: {final_role.value}
Original statement: "{original_statement}"

Rephrased statement (keep the same strategic meaning):"""
        
        try:
            output = self.llm(
                rephrase_prompt,
                max_tokens=100,
                temperature=0.7,
                stop=["\n", "\""]
            )
            
            # llama-cpp-python returns a dict with 'choices' key
            if isinstance(output, dict) and 'choices' in output:
                rephrased = output['choices'][0]['text'].strip()
                if rephrased and len(rephrased) > 10:  # Basic quality check
                    return rephrased
        except Exception as e:
            logger.warning(f"Failed to rephrase statement: {e}")
        
        return None
    
    async def generate_sft_dataset(self, num_games: int) -> List[SFTDataPoint]:
        """Generate a complete SFT dataset."""
        logger.info(f"Generating SFT dataset with {num_games} games")
        
        configs = self.generate_game_configs(num_games)
        all_data_points = []
        
        for i, (players, roles) in enumerate(configs):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{num_games} games")
            
            try:
                data_points = await self.simulate_game_for_sft(players, roles)
                all_data_points.extend(data_points)
            except Exception as e:
                logger.error(f"Failed to simulate game {i}: {e}")
                continue
        
        logger.info(f"Generated {len(all_data_points)} training examples")
        return all_data_points
    
    def save_dataset(self, data_points: List[SFTDataPoint], output_dir: Path):
        """Save the dataset in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL for easy loading
        jsonl_path = output_dir / "sft_dataset.jsonl"
        with open(jsonl_path, 'w') as f:
            for dp in data_points:
                json.dump(asdict(dp), f)
                f.write('\n')
        
        # Save as separate prompt/completion files for some training frameworks
        prompts_path = output_dir / "prompts.txt"
        completions_path = output_dir / "completions.txt"
        
        with open(prompts_path, 'w') as pf, open(completions_path, 'w') as cf:
            for dp in data_points:
                pf.write(dp.prompt + '\n---\n')
                cf.write(dp.completion + '\n---\n')
        
        # Save metadata summary
        metadata_path = output_dir / "metadata.json"
        metadata_summary = {
            "total_examples": len(data_points),
            "roles_distribution": {},
            "strategy_variants": {},
            "final_round_examples": 0,
        }
        
        for dp in data_points:
            role = dp.metadata["final_role"]
            variant = dp.metadata["strategy_variant"]
            
            metadata_summary["roles_distribution"][role] = metadata_summary["roles_distribution"].get(role, 0) + 1
            metadata_summary["strategy_variants"][variant] = metadata_summary["strategy_variants"].get(variant, 0) + 1
            
            if dp.metadata["is_final_round"]:
                metadata_summary["final_round_examples"] += 1
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_summary, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}")
        logger.info(f"Files created: {jsonl_path}, {prompts_path}, {completions_path}, {metadata_path}")


async def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data for Werewolf LLM")
    parser.add_argument("-m", "--model_path", type=str, help="Path to GGUF model for rephrasing (automatically enables LLM rephrasing)")
    parser.add_argument("-o", "--output_dir", type=str, default="./sft_data", help="Output directory for dataset")
    parser.add_argument("-n", "--num_games", type=int, default=1000, help="Number of games to simulate")
    
    args = parser.parse_args()
    
    # If model_path is provided, automatically enable LLM rephrasing
    if args.model_path:
        logger.info("Model path provided - automatically enabling LLM rephrasing")
    
    # Validate arguments
    if args.model_path and not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Create generator
    generator = SFTDataGenerator(model_path=args.model_path)
    
    try:
        # Generate dataset
        data_points = await generator.generate_sft_dataset(args.num_games)
        
        # Save dataset
        output_dir = Path(args.output_dir)
        generator.save_dataset(data_points, output_dir)
        
        logger.info("SFT data generation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


def main_sync():
    """Synchronous wrapper for the main function to work as a script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
