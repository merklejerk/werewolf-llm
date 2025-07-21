## **One Night Ultimate Werewolf LLM + RL Project Specification**

### **1. Project Overview**

This document details the technical specification for developing an intelligent agent capable of playing a simplified version of One Night Ultimate Werewolf (ONUW). The core methodology involves leveraging a small Large Language Model (LLM), such as Gemma 3 or Qwen 3, trained through a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) via self-play. The project's primary objective is to enable the LLM agent to exhibit strategic communication, deception, and deduction during the game's discussion phase, with performance evaluated based on team victory.

### **2\. System Architecture**

The proposed system architecture comprises several interconnected modules, primarily implemented in Python, utilizing the Hugging Face transformers and trl libraries.

```graph TD  
    A\[Game Simulator\] \--\> B{Data Generation Pipeline};  
    B \--\> C\[SFT Dataset\];  
    C \--\> D\[SFT Trainer\];  
    D \--\> E\[SFT-Tuned LLM Policy\];  
    E \-- Self-Play Episodes \--\> F\[RL Trainer (TRL/PPO)\];  
    F \--\> E;  
    F \-- Game Outcomes / Rewards \--\> A;  
    E \-- Actions (Statements, Votes, Guesses) \--\> A;  
    A \-- Observations (Game State) \--\> E;
```

**Components:**

* **Game Simulator:** A deterministic Python environment responsible for simulating ONUW game mechanics, including role assignment, simulated night actions, day phase progression (discussion), voting, and final resolution. It serves as the primary interface for providing environmental observations to agents and computing rewards.  
* **LLM Agent (Policy & Value Head):** A fine-tuned LLM (Gemma 3 / Qwen 3\) instance functioning as a game player. This LLM will incorporate a policy head for generating public statements and votes, and a value head for estimating state-value functions, crucial for the PPO algorithm. Additionally, it will generate private role guesses.  
* **Data Generation Pipeline:** A set of scripts designed to produce synthetic game logs for the SFT phase. This pipeline will utilize rule-based agents operating within the Game Simulator.  
* **SFT Dataset:** A curated collection of input-output pairs (prompts and corresponding desired responses) used to initialize the LLM's understanding of game mechanics and communication patterns.  
* **SFT Trainer:** A training module leveraging Hugging Face transformers.Trainer or trl.SFTTrainer to perform supervised fine-tuning on the SFT Dataset.  
* **RL Trainer (TRL/PPO):** A reinforcement learning module utilizing trl.PPOTrainer to train the LLM agents. This module implements the Proximal Policy Optimization (PPO) algorithm within a multi-agent self-play framework.

### **3\. Game Mechanics (Simplified ONUW)**

To manage computational and developmental complexity, a simplified version of ONUW will be implemented.

3.1. Roles:  
A small, fixed set of core ONUW roles will be utilized. Representative roles include:

* **Villager:** Possesses no special night action. Achieves victory if a Werewolf is successfully executed.  
* **Werewolf:** Identifies other Werewolves (if present). Achieves victory if no Werewolf is executed.  
* **Seer:** Observes the final role of one selected player or two selected center cards. Aligns with the Villager team's victory condition.  
* **Troublemaker:** Swaps the roles of two other players. Aligns with the Villager team's victory condition.  
* **Robber:** Swaps their own role with another player's role, subsequently observing their newly assigned role. Aligns with the Villager team's victory condition.  
* **Tanner:** Achieves victory exclusively if they are executed.

**3.2. Players & Center Cards:**

* A fixed number of players (e.g., 3-5 participants).  
* A fixed number of center cards (typically 3).  
* The total number of roles in play (assigned to players and center cards) will correspond to the sum of selected roles for the game instance.

**3.3. Game Phases:**

* **Simulated Night Phase (No LLM Actions):**  
  * The Game Simulator will randomly assign initial roles to all players and center cards.  
  * Night actions will then be simulated deterministically, or with controlled stochasticity for target selection (e.g., by Troublemaker or Robber), adhering to the standard ONUW night order.  
  * **Crucially:** The simulator will compute the *final role* of each player and center card after all night actions have been resolved.  
  * For roles that acquire information during the night (e.g., Seer, Werewolf, Robber, Insomniac), the simulator will generate and store the precise "observation" (e.g., "You observed Player X as \[Role\]", "Your new role is \[Role\]") that the respective player would receive.  
* **Day Phase (LLM Actions):**  
  * Each LLM agent will receive its *final role* and any *night observations* as part of its initial prompt for the day phase.  
  * Players will take turns making public statements. The number of discussion rounds can be configured (e.g., a single round or multiple).  
  * **LLM Action:** During its turn, each LLM agent will generate:  
    * A **public statement** in natural language.  
    * A **private guess** of all players' final roles, presented in a structured JSON format.
    * A **vote** for the player it believes to be a Werewolf.
* **Voting & Resolution Phase:**  
  * Following the final discussion round, the Game Simulator will use the `VOTE` from each player's *final* turn output. The `PUBLIC_STATEMENT` from this final turn is discarded.
  * The Game Simulator will tally the votes.  
  * The player with the most votes will be designated as executed.  
  * ONUW win conditions will be applied based on the executed player's *final role* and the *final roles* of all players and center cards.  
  * Rewards (+1 for team victory, -1 for team defeat) will be assigned to each LLM agent based on its team's outcome.  
  * Auxiliary rewards for private role guesses will be computed based on their accuracy against the simulator's ground truth.

### **4\. LLM Agent Design**

The LLM agent will be a fine-tuned instance of a small model (Gemma 3 or Qwen 3), configured for both natural language generation and structured output.

4.1. Input Prompt Structure:  
The prompt provided to each LLM agent will encapsulate all pertinent information for its decision-making turn. The structure is consistent for every turn. Example:  

```
[SYSTEM_INSTRUCTION] You are playing One Night Ultimate Werewolf. Your objective is to secure victory for your team. Formulate strategic statements and votes. Respond with a JSON object containing publicStatement, privateRoleGuesses, and vote fields.

[GAME_STATE]
Players: PlayerA, PlayerB, PlayerC, PlayerD
Possible Roles in Play: Villager, Werewolf, Seer, Troublemaker, Robber
Center Cards: [Card1, Card2, Card3] (hidden from players)

[YOUR_PLAYER_ID] PlayerB

[YOUR_INITIAL_ROLE] You are the Seer.
[YOUR_NIGHT_OBSERVATIONS] You observed PlayerB as a Werewolf.

[DISCUSSION_TRANSCRIPT]
PlayerA: "I am a Villager. I performed no night action."
PlayerC: "I am the Robber. I exchanged roles with PlayerD and am now a Villager."
```

4.2. Output Format (Structured JSON):  
The LLM will be trained to consistently produce responses in a strict JSON format with camelCase field names. This strict adherence is critical for reliable programmatic parsing by the Game Simulator. The `vote` field is required in every turn's output.

```
{  
  "publicStatement": "I am the Seer. I observed PlayerB as a Werewolf. PlayerB's statement is inconsistent with my observation.",  
  "privateRoleGuesses": {  
    "PlayerA": "Villager",  
    "PlayerB": "Werewolf",  
    "PlayerC": "Robber",  
    "PlayerD": "Villager",  
    "CenterCard1": "Tanner",  
    "CenterCard2": "Seer",  
    "CenterCard3": "Werewolf"  
  },  
  "vote": "PlayerB"  
}
```

4.3. Parameter-Efficient Fine-Tuning (PEFT) with LoRA:  
To optimize computational resources and enable efficient training on smaller hardware, Parameter-Efficient Fine-Tuning (PEFT) using the Low-Rank Adaptation (LoRA) method will be employed.

* **Mechanism:** LoRA introduces small, trainable low-rank matrices into the transformer's attention layers. During fine-tuning (both SFT and RL), only these LoRA adapter weights are updated, while the majority of the pre-trained LLM's parameters remain frozen.  
* **Benefits:**  
  * **Reduced VRAM Consumption:** Significantly lowers memory footprint, allowing for training larger models or larger batch sizes on consumer-grade GPUs.  
  * **Faster Training:** Fewer trainable parameters lead to quicker gradient computations and faster training iterations.  
  * **Smaller Checkpoints:** Only the compact LoRA adapter weights need to be saved, simplifying storage and deployment.  
  * **Performance Retention:** LoRA typically achieves performance comparable to full fine-tuning across various tasks.  
* **Integration:** The Hugging Face trl library provides native support for LoRA, enabling its seamless integration into both the SFT and PPO training pipelines for the LLM's policy (actor) and value head (critic).

### **5\. Data Generation (Supervised Fine-Tuning \- SFT Phase)**

This phase aims to provide a "warm-start" for the LLM, establishing foundational language comprehension, expected output formatting, and a baseline of plausible game-play behavior.

**5.1. Strategy for Generating Logs:**

* **Rule-Based Agents:** The primary methodology for generating SFT data will involve executing numerous simulations using simple, deterministic, or minimally randomized rule-based AI agents within your Game Simulator.  
  * **Mechanical Accuracy:** These agents will ensure that generated logs are mechanically accurate and logically consistent with the game rules (e.g., a Seer's observation will always correspond to the *final* role of the observed player). This approach prevents the model from being "poisoned" by contradictory game logic.  
  * **Plausible Behavior:** Heuristics will be defined for each role's statement and voting behavior.  
    * *Example (Villager):* "Claim Villager. Accuse players exhibiting suspicious behavior."  
    * *Example (Werewolf):* "Claim Villager or Seer. Accuse a non-Werewolf player."  
    * *Example (Seer):* "Claim Seer and truthfully state observation."  
  * **Diversity:** Controlled randomness will be introduced into rule-based agent behaviors (e.g., random target selection for roles like Seer/Robber/Troublemaker, random accusation targets among suspicious players).  
* **Optional: LLM Rephrasing for Linguistic Diversity:** If enhanced linguistic naturalness is desired, a larger, more general LLM (e.g., GPT-4o or Gemini 1.5 Pro) could be employed to *rephrase* statements generated by rule-based agents. This would be a post-processing step, requiring rigorous verification to prevent the introduction of logical inconsistencies. For initial project implementation, relying solely on rule-based agent generation is recommended for simplicity and robustness.

5.2. Data Format:  
Each entry in the SFT dataset will comprise a (prompt, completion) pair:

* **Prompt:** Comprising the [SYSTEM_INSTRUCTION], [GAME_STATE], [YOUR_PLAYER_ID], [YOUR_ROLE], [YOUR_NIGHT_OBSERVATIONS], and [DISCUSSION_TRANSCRIPT] elements as detailed in Section 4.1.  
* **Completion:** The desired structured JSON output with camelCase field names, including publicStatement, privateRoleGuesses, and vote.

**5.3. Quantity Considerations:**

* **Minimum Volume:** A target of several thousand (e.g., 5,000-10,000) high-quality (prompt, completion) pairs is recommended. Smaller LLMs generally benefit significantly from larger datasets.  
* **Diversity:** The dataset must encompass a wide array of game states, role combinations, and discussion flows to ensure comprehensive coverage.

### **6\. Reinforcement Learning (RL) Phase**

This phase is dedicated to training the SFT-tuned LLMs to optimize for team victory through self-play.

6.1. Objective:  
The primary objective is to maximize the expected cumulative reward, directly correlating with the team's win rate.  
**6.2. Reward Function:**

* **Primary Extrinsic Reward (Sparse):**  
  * \+1: Awarded if the agent's team achieves victory in the round.  
  * \-1: Assigned if the agent's team experiences defeat in the round.  
  * This reward is provided exclusively at the conclusion of each game episode.  
* **Auxiliary Intrinsic Reward (Dense):**  
  * **Private Role Guess Accuracy:** A small reward will be calculated on each turn when an agent outputs PRIVATE\_ROLE\_GUESSES. This reward will be based on the accuracy of its guesses against the Game Simulator's ground truth of *final roles*.  
    * *Calculation:* This can be a simple count of correctly identified roles, or a more sophisticated metric such as F1-score for imbalanced role distributions.  
    * *Weighting:* This auxiliary reward must be weighted significantly lower than the primary win/loss reward (e.g., 0.01 or 0.001 multiplied by the accuracy score). This ensures it serves as a beneficial signal for developing internal belief states without overriding the ultimate objective of winning the game. It can be integrated directly into the PPO loss function or used to shape the advantage function.

**6.3. Training Process:**

* **Multi-Agent Self-Play:** Multiple LLM agents (all instances derived from the fine-tuned model) will engage in adversarial play against each other. Each complete game constitutes an episode.  
* **PPO Algorithm (via TRL):**  
  * The LLM policy will be initialized with the weights from the SFT-tuned model.  
  * The trl.PPOTrainer will orchestrate the PPO training loop:  
    1. **Rollout:** Agents participate in a full game (or a batch of games). At each step, the LLM generates a public statement and private guesses/vote.  
    2. **Reward Collection:** The Game Simulator provides the sparse team win/loss reward and the dense auxiliary role guess reward.  
    3. **Advantage Estimation (GAE):** The PPO algorithm, leveraging the LLM's value head, computes the advantage for each action (statement, vote, guess) executed during the episode. This mechanism addresses the credit assignment problem.  
    4. **Policy Update:** The LLM's policy (actor) and value head (critic) parameters are updated using the PPO loss function to increase the probability of actions that yield higher advantages and ultimately, lead to team victories.  
* **Batching:** TRL facilitates efficient GPU utilization through effective batching of prompts and responses.  
* **Reference Model:** TRL's PPO typically employs a ref\_model (a frozen copy of the SFT-tuned model) to compute a KL divergence penalty. This penalty mitigates excessive divergence of the RL-trained model from the initial SFT policy, thereby preserving linguistic fluency and instruction following while enabling the acquisition of new strategies.

**6.4. Hyperparameters (General Guidance):**

* **Learning Rate:** Recommended range for experimentation: 1e-5 to 5e-6.  
* **PPO Epochs:** Typically 2-4 optimization epochs per PPO batch.  
* **Batch Size:** Constrained by GPU memory; gradient\_accumulation\_steps can be employed to simulate larger batch sizes.  
* **GAE Parameters (gamma, lambda):** Standard values (gamma=0.99, lambda=0.95) serve as robust initial settings.  
* **KL Coefficient (kl\_ctl in TRL):** This parameter regulates the degree of divergence permitted from the reference model. Initial experimentation should begin with a small value.  
* **Auxiliary Reward Weight:** Critical for fine-tuning. Start with a very low value (e.g., 0.01 or 0.001) and incrementally increase if the primary learning rate is too slow.

### **7\. Technical Stack**

* **Programming Language:** Python 3.x  
* **LLM Framework:** Hugging Face transformers  
* **Reinforcement Learning Library:** Hugging Face trl (specifically PPOTrainer)  
* **Deep Learning Framework:** PyTorch (serving as the backend for transformers and trl)  
* **Game Simulator:** Custom-developed Python classes and functions.  
* **Hardware Requirements:** Training LLMs with RL is computationally intensive. A dedicated GPU (e.g., NVIDIA RTX 3090/4090 or A100) is essential. Cloud computing resources (GCP, AWS, Azure) may be necessary for extensive training runs.

### **8\. Evaluation Metrics**

* **Primary Metric:**  
  * **Team Win Rate:** The percentage of games won by the agent's team, averaged across all roles.  
* **Secondary Metrics (for analysis and debugging):**  
  * **Role-Specific Win Rates:** Win rates disaggregated by individual roles (e.g., Werewolf, Villager, Tanner).  
  * **Private Role Guess Accuracy:** The average accuracy of the PRIVATE\_ROLE\_GUESSES output.  
  * **Statement Coherence/Plausibility:** Qualitative assessment, or potentially quantifiable using a separate LLM-based evaluator (though more complex to implement programmatically).  
  * **Voting Consistency:** Analysis of how frequently an agent's vote aligns with its stated beliefs or its team's optimal strategy.

### **9\. Potential Challenges & Future Work**

* **Computational Cost:** RL training is inherently slow. Anticipate extended training durations and potentially significant computational resource expenditure.  
* **Hyperparameter Sensitivity:** RL algorithms are highly sensitive to hyperparameter configurations. Extensive iterative experimentation and tuning will be required.  
* **Emergent Behavior Analysis:** Interpreting the underlying rationale for complex strategic decisions made by LLMs (particularly deceptive ones) will be a challenging but insightful endeavor.  
* **Scalability:** Expanding the project to accommodate a larger number of players or more intricate roles will exponentially increase the state and action space complexity.  
* **Human-AI Play:** A compelling future development involves enabling human players to participate in games alongside the trained LLM agents, facilitating evaluation of their human-likeness and strategic depth.  
* **Dynamic Prompting:** Investigating advanced prompt generation techniques during the RL phase, where prompts dynamically adapt based on the LLM's real-time performance or specific in-game events.

### **Appendix: Reasoning, Justifications, and Pitfalls**

This appendix summarizes the key discussions and decisions made during the development of this project specification, highlighting the rationale behind various choices and anticipating potential challenges.

#### **A. Choice of Game: One Night Ultimate Werewolf (ONUW)**

* **Justification:** ONUW was selected due to its inherent blend of language-based communication, social deduction, and hidden information, which aligns strongly with the capabilities of Large Language Models. Its single-round structure simplifies temporal credit assignment compared to multi-round games, while still presenting a complex strategic challenge.  
* **Pitfalls & Mitigations:**  
  * **Subjectivity of "Good Play":** Human interpretation of concepts like "good communication" or "effective deception" is inherently subjective and difficult to quantify directly.  
    * **Mitigation:** The primary reward function is designed to be strictly objective (team win/loss). The RL algorithm, through extensive self-play, will learn what constitutes "good play" in terms of achieving this objective, thereby circumventing the need for hand-crafted, subjective reward signals.  
  * **Hidden Information & Dynamic Roles:** The night phase, with its potential for role changes (e.g., via Troublemaker, Robber), introduces significant uncertainty. This dynamic nature renders simple "truth-telling" reward functions unviable.  
    * **Mitigation:** The "Simulated Night Phase" is implemented. The game simulator autonomously handles all night actions and computes the *final roles* and *observations*. LLM agents participate exclusively in the Day and Voting phases, receiving their *final* role and relevant observations as input. This approach simplifies the agent's state representation and eliminates the requirement for the LLM to learn complex night-phase mechanics.

#### **B. Reward Function Design**

* **Justification (Primary Reward):** A straightforward binary reward (+1 for team victory, \-1 for team defeat) is selected as the primary extrinsic reward. This represents the most objective and unambiguous signal for the ultimate game goal.  
* **Justification (Auxiliary Reward for Private Role Guesses):**  
  * **Denser Signal:** This auxiliary reward provides more frequent feedback compared to the sparse, end-of-game win/loss reward, potentially accelerating the learning process.  
  * **Improved Internal Belief State:** It encourages the LLM to actively engage in deduction and maintain a coherent internal model of other players' hidden roles, which is fundamental for strategic gameplay.  
  * **Addresses Partial Observability:** This mechanism assists the agent in reconstructing the complete game state from its limited, partially observable information.  
* **Pitfalls & Mitigations (Reward Hacking):**  
  * **Auxiliary Reward Over-optimization:** A risk exists wherein the LLM might prioritize maximizing its private role guess accuracy over its team's ultimate victory if the auxiliary reward's weighting is excessive or misaligned.  
    * **Mitigation:** The auxiliary reward will be **weighted significantly lower** (e.g., 0.01 or 0.001 relative to the primary reward). Its function is to serve as a "helper" signal, not the overriding objective. It can also be incorporated as an auxiliary loss term within the PPO objective function.  
  * **Truthfulness vs. Deception:** Directly rewarding "truthfulness" based on initial role assignments is problematic due to the dynamic nature of roles in ONUW.  
    * **Mitigation:** The auxiliary reward for private role guesses is based on accuracy against the *simulator's ground truth of final roles*. This approach inherently accounts for night-phase role changes and rewards the agent for accurate deduction, irrespective of its public communication strategy.  
  * **No Direct "Lying" Reward:** The system design explicitly *avoids* a direct reward for "effective lying." Instead, effective deceptive strategies are expected to emerge organically as they contribute to the ultimate team victory, which is the sole focus of the primary reward.

#### **C. Training Process: SFT followed by RL**

* **Justification (SFT Warm-up):**  
  * **Structured Output Compliance:** Smaller LLMs often exhibit challenges in consistently generating complex structured outputs (e.g., JSON) without specific fine-tuning. SFT will train the LLM to reliably produce the required format, thereby preventing parsing errors during the subsequent RL phase.  
  * **Foundational Language & Game Understanding:** SFT on simulated game logs will imbue the LLM with the domain-specific vocabulary, phrasing, and fundamental patterns of ONUW discussions (e.g., typical role claims, accusation styles, defensive maneuvers). This provides a "plausible" initial policy.  
  * **Improved RL Efficiency & Stability:** Commencing RL training from a well-behaved SFT-tuned model (a "warm start") is considerably more efficient and stable than training from a randomly initialized policy, significantly reducing the vast exploration space.  
* **Justification (RL for Strategy & Generalization):**  
  * **Outcome-Based Learning:** RL is indispensable for optimizing the LLM's behavior based on actual game outcomes (wins/losses). This enables the LLM to learn complex, emergent strategies (including sophisticated deception and deduction) that directly lead to victory.  
  * **Generalization:** While SFT primarily facilitates memorization of training patterns, RL is superior in generalizing learned strategies to novel game states and unforeseen opponent behaviors.  
  * **Credit Assignment:** The PPO algorithm's utilization of a value head and Generalized Advantage Estimation (GAE) effectively addresses the temporal credit assignment problem, allowing the model to discern which specific actions contributed to the final reward, even when that reward is delayed.  
* **Pitfalls & Mitigations (SFT Data Quality):**  
  * **Inconsistent SFT Data:** Employing a large, general-purpose LLM to generate SFT logs without stringent mechanical accuracy can potentially "poison" the model with illogical or contradictory game scenarios.  
    * **Mitigation:** The SFT data will be predominantly generated by **rule-based agents** operating within the Game Simulator. These agents ensure the mechanical accuracy and logical consistency of the generated game logs, even if their strategic depth is limited. This approach establishes a clean, consistent foundation for the LLM's initial learning. The optional use of a larger LLM for linguistic diversity would be a separate, carefully controlled post-processing step.  
  * **Suboptimal SFT Policy:** The initial SFT policy may not be optimal.  
    * **Mitigation:** This is an anticipated outcome. The RL phase is specifically designed to take this plausible but suboptimal policy and iteratively refine it into an optimal, winning strategy through continuous self-play.

#### **D. Training Paradigm: Multi-Agent Self-Play**

* **Justification:**  
  * **Implicit Coordination & Deception:** Self-play represents the most potent method for training agents in adversarial, hidden-information games. It naturally fosters the emergence of robust strategies, including sophisticated deception and counter-deception, as agents continuously adapt to intelligent, evolving opponents (which are other instances of themselves).  
  * **Scalability:** This paradigm facilitates the generation of vast quantities of training data without requiring extensive human labeling or expert demonstrations for every game state.  
  * **Robustness:** Agents trained through self-play tend to exhibit greater robustness against a diverse range of opponent strategies.  
* **Pitfalls:**  
  * **Computational Cost:** Executing numerous simultaneous LLM inference calls within a self-play loop is computationally intensive.  
  * **Non-Stationarity:** In Multi-Agent Reinforcement Learning (MARL), the environment is inherently non-stationary because all agents are concurrently learning and modifying their policies. This dynamic can complicate convergence.  
    * **Mitigation:** PPO, with its trust-region approach and KL divergence penalty (implemented via the ref\_model in TRL), contributes to training stability by constraining the magnitude of policy updates.

#### **E. Parameter-Efficient Fine-Tuning (PEFT) with LoRA**

* **Justification:** LoRA is chosen to optimize computational resources and enable efficient training on hardware with limited memory, such as consumer-grade GPUs.  
* **Mechanism:** LoRA introduces small, trainable low-rank matrices into the transformer's attention layers. During both SFT and RL, only these LoRA adapter weights are updated, while the majority of the pre-trained LLM's parameters remain frozen.  
* **Benefits:**  
  * **Reduced VRAM Consumption:** Significantly lowers memory footprint, enabling fine-tuning of larger models or the use of larger effective batch sizes on more modest hardware.  
  * **Faster Training:** The reduced number of trainable parameters accelerates gradient computations and overall training iterations.  
  * **Smaller Storage Footprint:** Only the compact LoRA adapter weights (typically in megabytes) need to be stored, simplifying checkpointing, distribution, and deployment.  
  * **Comparable Performance:** LoRA fine-tuning often achieves performance highly comparable to full fine-tuning across various downstream tasks.  
* **Integration:** The Hugging Face trl library provides native support for PEFT methods like LoRA, ensuring straightforward integration into both the SFT and PPO training pipelines for the LLM's policy (actor) and value head (critic).