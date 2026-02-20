import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import math
from collections import Counter

directions = [
	(-1, 0), (1, 0), (0, -1), (0, 1),  # left, right, down, up
	(-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
]

class Agent:
	def __init__(self, x, y, sex, color, energy=10):
		self.x = x
		self.y = y
		self.sex = sex
		self.color = color
		self.energy = energy

class Food:
	def __init__(self, x, y, color, energy):
		self.x = x
		self.y = y
		self.color = color
		self.energy = energy

class Apple(Food):
	def __init__(self, x, y):
		super().__init__(x, y, color=(1, 0, 0), energy=5)  # red

class Orange(Food):
	def __init__(self, x, y):
		super().__init__(x, y, color=(1, 0.5, 0), energy=10)  # orange

class Wall:
	def __init__(self, x, y, color=(0, 1, 0)):  # green
		self.x = x
		self.y = y
		self.color = color

class Grid:
	def __init__(self, width, height, metabolic_cost, min_child_energy, reproduction_cost, food_respawn_rate,
	num_agents, num_apples, num_oranges, num_walls=30, use_nn=False, seed=123):
		self.width = width
		self.height = height
		self.metabolic_cost = metabolic_cost
		self.min_child_energy = min_child_energy
		self.reproduction_cost = reproduction_cost
		self.food_respawn_rate = food_respawn_rate
		self.grid = [[None for _ in range(width)] for _ in range(height)]
		self.agents = []
		self.food_items = []
		self.walls = []
		self.use_nn = use_nn
		self.rng = random.Random(seed)

		self.populate(num_agents, num_apples, num_oranges, num_walls)

	def get_empty_positions(self):
		return [(i, j) for i in range(self.height) for j in range(self.width) if self.grid[i][j] is None]

	def is_empty(self, x, y):
		return (0 <= x < self.height) and (0 <= y < self.width) and self.grid[x][y] is None

	def place_object(self, obj):
		self.grid[obj.x][obj.y] = obj

	def remove_object(self, obj):
		if self.grid[obj.x][obj.y] is obj:
			self.grid[obj.x][obj.y] = None

	def place_wall(self):
		directions = [(1, 0), (0, 1)]  # vertical, horizontal
		start_positions = self.get_empty_positions()
		if not start_positions:
			return

		start_x, start_y = self.rng.choice(start_positions)
		dir_x, dir_y = self.rng.choice(directions)
		length = self.rng.randint(2, 6)

		for i in range(length):
			nx, ny = start_x + i * dir_x, start_y + i * dir_y
			if not self.is_empty(nx, ny):
				break
			wall = Wall(nx, ny)
			self.place_object(wall)
			self.walls.append(wall)

	def populate(self, num_agents, num_apples, num_oranges, num_walls):
		for _ in range(num_walls):
			self.place_wall()

		total_needed = num_agents + num_apples + num_oranges
		empty_positions = self.get_empty_positions()

		if total_needed > len(empty_positions):
			raise ValueError("Not enough space to place all entities.")

		selected_positions = self.rng.sample(empty_positions, total_needed)
		index = 0

		for _ in range(num_agents):
			x, y = selected_positions[index]
			sex = self.rng.randint(0, 1)
			color = (0, 0, 1) if sex == 0 else (0, 0, 0.5)
			agent = Agent(x, y, sex, color)

			# Placeholder for NN
			if self.use_nn:
				agent.nn = initialize_new_nn()

			self.place_object(agent)
			self.agents.append(agent)
			index += 1

		for _ in range(num_apples):
			x, y = selected_positions[index]
			apple = Apple(x, y)
			self.place_object(apple)
			self.food_items.append(apple)
			index += 1

		for _ in range(num_oranges):
			x, y = selected_positions[index]
			orange = Orange(x, y)
			self.place_object(orange)
			self.food_items.append(orange)
			index += 1

	def meet(self, agent1, agent2):
		if agent1 not in self.agents or agent2 not in self.agents:
			return

		# Fighting scenario
		if agent1.sex == agent2.sex:
			if agent1.energy > agent2.energy:
				winner, loser = agent1, agent2
			elif agent1.energy < agent2.energy:
				winner, loser = agent2, agent1
			else:
				winner, loser = (agent1, agent2) if self.rng.random() < 0.5 else (agent2, agent1)

			winner.energy += loser.energy
			self.remove_object(loser)
			self.agents.remove(loser)
			return

		# Mating scenario
		total_energy = agent1.energy + agent2.energy

		min_child_energy = self.min_child_energy
		reproduction_cost = self.reproduction_cost

		agent1.energy -= reproduction_cost
		agent2.energy -= reproduction_cost

		max_children = int(total_energy // min_child_energy)

		if max_children == 0:
			return

		# Center around mid-point between parents
		cx = (agent1.x + agent2.x) // 2
		cy = (agent1.y + agent2.y) // 2

		R = 2  # reproduction radius
		possible_spots = [
			(cx + dx, cy + dy)
			for dx in range(-R, R + 1)
			for dy in range(-R, R + 1)
			if not (dx == 0 and dy == 0)
			and self.is_empty(cx + dx, cy + dy)
		]
		self.rng.shuffle(possible_spots)

		num_children = min(len(possible_spots), max_children)
		
		if num_children == 0:
			return

		energy_per_child = total_energy / num_children

		children_positions = possible_spots[:num_children]

		for pos in children_positions:
			x, y = pos
			sex = self.rng.randint(0, 1)
			color = (0, 0, 1) if sex == 0 else (0, 0, 0.5)
			child = Agent(x, y, sex, color, energy=energy_per_child)

			# Placeholder for NN
			if self.use_nn:
				child.nn = cross_mutate(agent1.nn, agent2.nn)

			self.place_object(child)
			self.agents.append(child)

		# Remove parents
		for parent in [agent1, agent2]:
			self.remove_object(parent)
			self.agents.remove(parent)

	def get_agent_vision(self, agent, range_ext=2):
		"""
		Mixed vision:
		- Immediate 8-cell Moore neighborhood
		- Cardinal extensions (N, S, E, W) up to range_ext

		Returns:
			np.ndarray of shape ((8 + 4*range_ext), 3)
		"""

		vision = []

		# --- Moore neighborhood ---
		for dx, dy in directions:
			nx, ny = agent.x + dx, agent.y + dy

			if not (0 <= nx < self.height and 0 <= ny < self.width):
				vision.append((0, 0, 0))
			else:
				obj = self.grid[nx][ny]
				if obj is None:
					vision.append((1, 1, 1))
				elif hasattr(obj, "color"):
					vision.append(obj.color)
				else:
					vision.append((0, 0, 0))

		# --- Cardinal extensions ---
		cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

		for dx, dy in cardinal_dirs:
			for r in range(1, range_ext + 1):
				nx, ny = agent.x + dx * r, agent.y + dy * r

				if not (0 <= nx < self.height and 0 <= ny < self.width):
					vision.append((0, 0, 0))
				else:
					obj = self.grid[nx][ny]
					if obj is None:
						vision.append((1, 1, 1))
					elif hasattr(obj, "color"):
						vision.append(obj.color)
					else:
						vision.append((0, 0, 0))

		return np.array(vision)

	def move_agent(self):

		self.rng.shuffle(self.agents)  # random move order

		for agent in list(self.agents):  # copy to avoid iteration issues
			if agent not in self.agents:
				return

			# Decay mechanism
			agent.energy -= self.metabolic_cost
			if agent.energy <= 0:
				self.remove_object(agent)
				self.agents.remove(agent)
				continue

			# Placeholder for NN
			if self.use_nn:
				vision = self.get_agent_vision(agent)
				agent.vision = vision

				dx, dy = agent.nn_output_to_move()
				pass
			else:
				# Random movement (default behavior)
				dx, dy = self.rng.choice(directions)

			nx, ny = agent.x + dx, agent.y + dy

			# Check bounds
			if not (0 <= nx < self.height and 0 <= ny < self.width):
				continue

			# Blocked by wall
			if isinstance(self.grid[nx][ny], Wall):
				continue

			target = self.grid[nx][ny]

			# Eat food
			if isinstance(target, Food):
				agent.energy += target.energy
				self.food_items.remove(target)
				self.remove_object(target)

			# Meet with another agent
			elif isinstance(target, Agent):
				self.meet(agent, target)

			# Check if agent is still alive
			if agent not in self.agents:
				continue

			# Move agent
			self.remove_object(agent)
			agent.x, agent.y = nx, ny
			self.place_object(agent)

	def respawn_food(self):
		empty = self.get_empty_positions()
		if not empty:
			return

		n = int(len(empty) * self.food_respawn_rate)
		for x, y in self.rng.sample(empty, n):
			if self.rng.random() < 0.6:
				food = Apple(x, y)
			else:
				food = Orange(x, y)

			self.place_object(food)
			self.food_items.append(food)

	def render(self):
		color_grid = np.ones((self.height, self.width, 3))  # white background

		for wall in self.walls:
			color_grid[wall.x][wall.y] = wall.color

		for agent in self.agents:
			color_grid[agent.x][agent.y] = agent.color

		for food in self.food_items:
			color_grid[food.x][food.y] = food.color

		num_apples = sum(isinstance(f, Apple) for f in self.food_items)
		num_oranges = sum(isinstance(f, Orange) for f in self.food_items)

		combined_energy = sum(agent.energy for agent in self.agents)

		title = (
			f"Agents: {len(self.agents)} | "
			f"Total agents energy: {combined_energy:.2f} | "
			f"Apples: {num_apples} | "
			f"Oranges: {num_oranges} | "
			f"Walls: {len(self.walls)}"
		)

		plt.figure(figsize=(10, 10))
		plt.imshow(color_grid, interpolation='nearest')
		plt.title(title)
		plt.axis('off')
		plt.show()


def set_seed(seed):
	"""
	Set the random seed across all relevant modules to ensure deterministic behavior.

	Parameters
	----------
	seed : int
		The seed value used to initialize Python's `random`, NumPy's random generator,
		and environment hash functions for reproducible simulation results.
	"""
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


def shannon_entropy(grid, energy_eps=6):
	"""
	Compute the coarse-grained Shannon entropy of the grid state.

	This entropy is a macroscopic, discretized classification of the system.
	Agents are grouped into categorical states based on sex and an energy
	threshold (HighE / LowE). Food types, walls, and empty cells are also
	classified discretely.

	Interpretation
	--------------
	- Entropy → discretized macroscopic classification
	- Lyapunov exponent → continuous phase-space metric

	This means entropy measures structural diversity at a coarse ecological
	level, while the Lyapunov exponent measures microscopic trajectory
	sensitivity in continuous state space.

	The entropy is computed as:

		H = -Σ (p_i * log2(p_i))

	where p_i is the probability of each discrete cell state.

	Returns
	-------
	float
		Shannon entropy of the discretized grid configuration.
	"""
	states = []
	for row in grid.grid:
		for cell in row:
			if cell is None:
				states.append("Empty")
			elif isinstance(cell, Agent):
				sex = "M" if cell.sex == 0 else "F"
				energy_class = "HighE" if cell.energy >= energy_eps else "LowE"

				states.append(f"Agent_{sex}_{energy_class}")
			elif isinstance(cell, Apple):
				states.append("Apple")
			elif isinstance(cell, Orange):
				states.append("Orange")
			elif isinstance(cell, Wall):
				states.append("Wall")

	counts = Counter(states)
	total = sum(counts.values())
	probs = [c / total for c in counts.values()]
	entropy = -sum(p * math.log2(p) for p in probs if p > 0)
	return entropy


def grid_difference(g1, g2):
	"""
	Compute the normalized continuous structural difference between two grids.

	This metric operates in continuous phase space:
	- Agent energy differences are treated continuously (normalized magnitude).
	- Sex mismatch contributes discretely.
	- Entity type mismatch contributes discretely.

	This is intentionally different from Shannon entropy:

	- Lyapunov exponent → continuous phase-space metric
	- Shannon entropy → discretized macroscopic classification

	The returned value represents a normalized phase-space distance
	between two system configurations.

	Returns
	-------
	float
		Normalized difference in the range [0, 1] (approximately).
	"""
	diff = 0.0
	total = g1.width * g1.height

	for i in range(g1.height):
		for j in range(g1.width):
			c1, c2 = g1.grid[i][j], g2.grid[i][j]

			if (c1 is None) != (c2 is None):
				diff += 1.0

			elif (c1 is not None) and (c2 is not None):

				if type(c1) != type(c2):
					diff += 1.0

				elif isinstance(c1, Agent):
					# continuous energy difference
					energy_scale = max(abs(c1.energy), abs(c2.energy), 1.0)
					energy_term = abs(c1.energy - c2.energy) / energy_scale

					sex_term = 0.0 if c1.sex == c2.sex else 1.0

					diff += 0.5 * sex_term + 0.5 * energy_term

	return diff / total


def lyapunov_analysis(g1, g2, num_ticks=50, render=False, **grid_params):
	"""
	Estimate the maximal Lyapunov exponent via lockstep evolution.

	The two grids are evolved in strict lockstep per tick to ensure
	dynamical comparability. At each tick, their continuous phase-space
	distance is measured using `grid_difference`.

	The exponent is estimated by fitting a linear slope to the early
	linear-growth region of log(d(t)).

	Conceptual distinction:
	- Lyapunov exponent → continuous phase-space instability metric
	- Shannon entropy → coarse-grained macroscopic classification

	This function estimates microscopic dynamical sensitivity,
	not macroscopic disorder.

	Returns
	-------
	tuple of (float, list)
		Estimated Lyapunov exponent and divergence trajectory.
	"""
	diffs = []

	# initial distance
	d0 = max(grid_difference(g1, g2), 1e-12)
	diffs.append(d0)

	if render == True:
		g1.render()
		print("Grid 1 - Initial Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Initial Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	for t in range(num_ticks):
		g1.move_agent()
		g1.respawn_food()

		# preserve RNG stream
		g2.move_agent()
		g2.respawn_food()

		d = max(grid_difference(g1, g2), 1e-12)
		diffs.append(d)

	if render == True:
		g1.render()
		print("Grid 1 - Final Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Final Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	# fit slope in early linear region
	log_diffs = np.log(diffs)

	# use 30% region before saturation
	cutoff = max(5, int(0.3 * len(log_diffs)))

	x = np.arange(cutoff)
	y = log_diffs[:cutoff]

	slope, _ = np.polyfit(x, y, 1)
	lyap = slope

	return lyap, diffs


def compare_grids(num_ticks=50, num_perturbed_agents=1, seed=123, final_render=True, lyapunov_final_render=True, num_trials=30, **grid_params):
	"""
	Compare two nearly identical grid simulations to estimate the Lyapunov exponent.

	A second grid is created as a deep copy of the first, with a small spatial
	perturbation applied to it. Both simulations are then evolved and
	compared over time to measure the divergence.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to run. Default is 50.
	num_perturbed_agents : int, optional
		Number of agents to be perturbated (by adding energy).
	seed : int, optional
		Random seed ensuring deterministic setup. Default is 123.

	Returns
	-------
	float
		Estimated Lyapunov exponent of the system.
	"""
	lambdas = []

	for trial in range(num_trials):
		#set_seed(seed + trial)
		g1 = Grid(**grid_params, seed=seed + trial)
		g2 = Grid(**grid_params, seed=seed + trial)

		if g2.agents:
			local_rng = random.Random(seed + 999 + trial)
			k = min(num_perturbed_agents, len(g2.agents))
			perturbed = local_rng.sample(g2.agents, k)

			for agent in perturbed:
				agent.energy += g2.min_child_energy

		lyap, diffs = lyapunov_analysis(
			g1, g2, num_ticks, lyapunov_final_render, **grid_params
		)

		lambdas.append(lyap)

	mean_lambda = np.mean(lambdas)
	std_lambda = np.std(lambdas)

	print(f"Lyapunov exponent: {mean_lambda:.6f} ± {std_lambda:.6f}")

	return mean_lambda


def check_determinism(num_ticks, seed, debug_render=False, final_render=True, **grid_params):
	"""
	Verify whether the simulation environment is fully deterministic.

	Runs two simulations with identical initial conditions and checks
	whether the final grid states are identical.

	Returns
	-------
	bool
		True if deterministic, False otherwise.
	"""
	print("----- DETERMINISTIC CHECK -----")

	#set_seed(seed)
	g1 = Grid(**grid_params, seed=seed)
	g2 = Grid(**grid_params, seed=seed)

	if final_render == True:
		g1.render()
		print("Grid 1 - Initial Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Initial Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	for t in range(num_ticks):
		# preserve RNG stream
		state_np = np.random.get_state()
		state_py = random.getstate()

		g1.move_agent()
		g1.respawn_food()

		np.random.set_state(state_np)
		random.setstate(state_py)

		# preserve RNG stream
		g2.move_agent()
		g2.respawn_food()
		
	if final_render == True:
		g1.render()
		print("Grid 1 - Final Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Final Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	diff = grid_difference(g1, g2)

	print(
		"Difference between two grids with identical initial conditions "
		f"(expected 0.0): {diff}"
	)

	if diff > 0:
		print("Determinism check FAILED.")
		return False

	print("Determinism check PASSED.")
	print("----- END OF DETERMINISTIC CHECK -----")
	return True


def main_simulation(num_ticks=50, num_perturbed_agents=1, seed=123, debug_render=False, final_render=True, lyapunov_final_render=True, num_trials=30, **grid_params):
	"""
	Execute the main experiment pipeline.

	This function:
	1. Runs a single grid simulation to observe entropy evolution.
	2. Runs a Lyapunov analysis to estimate system sensitivity to perturbations.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to run. Default is 50.
	num_perturbed_agents : int, optional
		Number of agents to be perturbated (by adding energy).
	seed : int, optional
		Random seed to ensure deterministic behavior. Default is 123.

	Returns
	-------
	tuple of (Grid, float)
		Final grid state and estimated Lyapunov exponent.
	"""

	is_deterministic = check_determinism(num_ticks=num_ticks,seed=seed,debug_render=debug_render,final_render=final_render,**grid_params)

	if not is_deterministic:
		raise RuntimeError("Simulation environment is non-deterministic.")

	print("----- LYAPUNOV EXPONENT COMPARISON -----")
	lyap = compare_grids(num_ticks, num_perturbed_agents, seed, final_render, lyapunov_final_render, num_trials, **grid_params)
	print("Estimated Lyapunov exponent:", lyap)
	print("----- END OF LYAPUNOV EXPONENT COMPARISON -----")



# Main Execution
grid_params = {
	"width": 50,
	"height": 50,
	"metabolic_cost":0.9,
	"min_child_energy": 7,
	"reproduction_cost": 5,
	"food_respawn_rate": 0.02,
	"num_agents": 20,
	"num_apples": 40,
	"num_oranges": 30,
	"num_walls": 60,
	"use_nn": False
}

main_simulation(
	num_ticks=1000,
	num_perturbed_agents=1,
	seed=123,
	debug_render=False, 
	final_render=True, 
	lyapunov_final_render=True, 
	num_trials=1,
	**grid_params
)