from werewolf_llm.game_simulator import GameSimulator
from werewolf_llm.roles import Role
from werewolf_llm.agent import Agent
import json
import asyncio


async def main():
    """
    Main function to run a single game of One Night Ultimate Werewolf.
    """
    players = ["Alice", "Bob", "Charlie", "David", "Eva", "Frank"]
    # Roles for 6 players + 3 center cards
    roles = [
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.SEER,
        Role.ROBBER,
        Role.TROUBLEMAKER,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.TANNER,
    ]

    print("--- Starting One Night Ultimate Werewolf Game ---")
    
    # 1. Setup and Night Phase
    game_sim = GameSimulator(players, roles)
    game_sim.run_game()

    print("\n--- Initial Roles (for debugging) ---")
    print(json.dumps({p: r.value for p, r in game_sim.initial_roles.items()}, indent=2))
    
    print("\n--- Night Observations (what each player knows) ---")
    print(json.dumps(game_sim.night_observations, indent=2))

    print("\n--- Final Roles (for debugging, after night actions) ---")
    print(json.dumps({p: r.value for p, r in game_sim.final_roles.items()}, indent=2))

    # 2. Day Phase (Discussion)
    print("\n--- Day Phase: Discussion ---")
    agents = {name: Agent(name) for name in players}
    
    # Single round of discussion
    for player_name in players:
        agent = agents[player_name]
        turn_output_str = await agent.generate_turn_output(game_sim, voting_phase=False)
        turn_output = json.loads(turn_output_str)
        statement = turn_output['PUBLIC_STATEMENT']
        print(f"{player_name}: {statement}")
        game_sim.add_to_discussion_transcript(player_name, statement)

    # 3. Voting Phase
    print("\n--- Voting Phase ---")
    votes = {}
    vote_tasks = [agents[name].get_vote(game_sim) for name in players]
    vote_results = await asyncio.gather(*vote_tasks)
    
    for i, player_name in enumerate(players):
        vote = vote_results[i]
        votes[player_name] = vote
        print(f"{player_name} votes for {vote}")

    # 4. Resolution
    executed_player, vote_counts = game_sim.resolve_voting(votes)
    print("\n--- Vote Results ---")
    print(json.dumps(vote_counts, indent=2))
    if executed_player:
        print(f"Player {executed_player} has been executed.")
    else:
        print("No one was executed due to a tie.")

    # 5. Determine Winner
    rewards = game_sim.determine_winner(executed_player)
    print("\n--- Game Over: Rewards ---")
    print(json.dumps(rewards, indent=2))

    print("\n--- Winning Team ---")
    # A bit of logic to determine the winning team for display
    if rewards[players[0]] == 1.0: # Check one player's reward
        # This logic is a bit simplistic, assumes uniform rewards for a team
        first_player_role = game_sim.final_roles[players[0]]
        if first_player_role == Role.WEREWOLF and any(r == Role.WEREWOLF for r in game_sim.final_roles.values()):
             print("Werewolf Team Wins!")
        elif first_player_role == Role.TANNER and executed_player and game_sim.final_roles.get(executed_player) == Role.TANNER:
             print("Tanner Wins!")
        else:
             print("Village Team Wins!")
    else:
        first_player_role = game_sim.final_roles[players[0]]
        if first_player_role != Role.WEREWOLF and any(r == Role.WEREWOLF for r in game_sim.final_roles.values()):
             print("Werewolf Team Wins!")
        elif first_player_role != Role.TANNER and executed_player and game_sim.final_roles.get(executed_player) == Role.TANNER:
             print("Tanner Wins!")
        else:
             print("Village Team Wins!")


if __name__ == "__main__":
    asyncio.run(main())
