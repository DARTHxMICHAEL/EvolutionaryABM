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
	num_agents, num_apples, num_oranges, num_walls=30, use_nn=False):
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

		start_x, start_y = random.choice(start_positions)
		dir_x, dir_y = random.choice(directions)
		length = random.randint(2, 6)

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

		selected_positions = random.sample(empty_positions, total_needed)
		index = 0

		for _ in range(num_agents):
			x, y = selected_positions[index]
			sex = random.randint(0, 1)
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
				winner, loser = (agent1, agent2) if random.random() < 0.5 else (agent2, agent1)

			winner.energy += loser.energy
			self.remove_object(loser)
			self.agents.remove(loser)
			return

		# Mating scenario
		total_energy = agent1.energy + agent2.energy

		min_child_energy = 6
		reproduction_cost = 4

		agent1.energy -= reproduction_cost
		agent2.energy -= reproduction_cost

		max_children = int(total_energy // min_child_energy)

		if max_children == 0:
			return

		# Center around mid-point between parents
		cx = (agent1.x + agent2.x) // 2
		cy = (agent1.y + agent2.y) // 2

		possible_spots = [
			(cx + dx, cy + dy) 
			for dx, dy in directions 
			if self.is_empty(cx + dx, cy + dy)
		]
		random.shuffle(possible_spots)

		num_children = min(len(possible_spots), max_children)
		
		if num_children == 0:
			return

		energy_per_child = total_energy / num_children

		children_positions = possible_spots[:num_children]

		for pos in children_positions:
			x, y = pos
			sex = random.randint(0, 1)
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

	def get_agent_vision(self, agent):
		"""
		Get RGB color data from the 8 surrounding cells (Moore neighborhood).
		Returns a numpy array of shape (8, 3) where each row is (R, G, B).

		Out-of-bounds cells → black (0, 0, 0)
		Empty cells → white (1, 1, 1)
		"""
		vision = np.zeros((8, 3))  # default black for all 8 directions

		for i, (dx, dy) in enumerate(directions):
			nx, ny = agent.x + dx, agent.y + dy

			# Out of bounds check
			if not (0 <= nx < self.height and 0 <= ny < self.width):
				vision[i] = (0, 0, 0)
				continue

			obj = self.grid[nx][ny]

			if obj is None:
				vision[i] = (1, 1, 1)
			elif hasattr(obj, "color"):
				vision[i] = obj.color
			else:
				vision[i] = (0, 0, 0)

		return vision

	def move_agent(self):

		random.shuffle(self.agents)  # random move order

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
				dx, dy = random.choice(directions)

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
		for x, y in random.sample(empty, n):
			if random.random() < 0.6:
				food = Apple(x, y)
			else:
				food = Orange(x, y)

			self.place_object(food)
			self.food_items.append(food)

	def run_simulation(self, ticks=1, render=False):
		for tick in range(ticks):
			self.move_agent()
			self.respawn_food()

			if render:
				self.render()
				print("Iteration: ", tick+1)

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

		title = (
			f"Agents: {len(self.agents)} | "
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

def shannon_entropy(grid):
	"""
	Compute the Shannon entropy of the grid's cell states.

	The entropy quantifies the distribution uniformity of cell types (e.g., agents,
	apples, oranges, walls, and empty cells) using the formula:

		H = -Σ (p_i * log2(p_i))

	where p_i is the probability of each cell state.

	Parameters
	----------
	grid : Grid
		The grid object containing cells with different entity types.

	Returns
	-------
	float
		The Shannon entropy value of the grid state distribution.
	"""
	states = []
	for row in grid.grid:
		for cell in row:
			if cell is None:
				states.append("Empty")
			elif isinstance(cell, Agent):
				states.append("Agent")
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
	Compute the normalized structural difference between two grids.

	The difference is calculated as the fraction of cells that differ
	in content (entity type or occupancy) between the two grids.

	Parameters
	----------
	g1, g2 : Grid
		Two grid instances of identical dimensions to compare.

	Returns
	-------
	float
		Normalized difference between grids in the range [0, 1].
	"""
	diff = 0
	total = g1.width * g1.height
	for i in range(g1.height):
		for j in range(g1.width):
			c1, c2 = g1.grid[i][j], g2.grid[i][j]
			if (c1 is None) != (c2 is None):
				diff += 1
			elif (c1 is not None) and (c2 is not None):
				if type(c1) != type(c2):  # different entity types
					diff += 1
	return diff / total

def lyapunov_analysis(g1, g2, num_ticks=50, render=False, final_render=True):
	"""
	Estimate the Lyapunov exponent from two initially similar grid simulations.

	Both grids evolve independently for a number of simulation ticks.
	The growth of their difference over time is used to estimate the
	Lyapunov exponent:

		λ = (1 / T) * ln(d_T / d_0)

	where d_T and d_0 are final and initial normalized grid differences.

	Parameters
	----------
	g1, g2 : Grid
		Two grid instances initialized with a small perturbation between them.
	num_ticks : int, optional
		Number of simulation steps to run for each grid. Default is 50.

	Returns
	-------
	tuple of (float, list)
		The estimated Lyapunov exponent and the list of difference values per tick.
	"""
	d0 = max(grid_difference(g1, g2), 1e-6)
	diffs = [d0]

	state = np.random.get_state()
	random_state = random.getstate()

	print("Initial Shannon entropy:", shannon_entropy(g1))

	g1.run_simulation(num_ticks, render)
	g1.render() if final_render == True else None

	print("Final Shannon entropy:", shannon_entropy(g1))
	
	np.random.set_state(state)
	random.setstate(random_state)

	print("Initial Shannon entropy:", shannon_entropy(g2))

	g2.run_simulation(num_ticks, render)
	g2.render() if final_render == True else None

	print("Final Shannon entropy:", shannon_entropy(g2))

	d = grid_difference(g1, g2)
	diffs.append(d)

	final_diff = max(diffs[-1], 1e-6)
	lyap = (1 / num_ticks) * math.log(final_diff / d0)

	return lyap, diffs

def compare_grids(num_ticks=50, perturbation=(0,1), seed=123, render=False, final_render=True, **grid_params):
	"""
	Compare two nearly identical grid simulations to estimate the Lyapunov exponent.

	A second grid is created as a deep copy of the first, with a small spatial
	perturbation applied to one agent. Both simulations are then evolved and
	compared over time to measure divergence.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to run. Default is 50.
	perturbation : tuple of (int, int), optional
		Offset (dx, dy) used to slightly move one agent in the copied grid.
	seed : int, optional
		Random seed ensuring deterministic setup. Default is 123.

	Returns
	-------
	float
		Estimated Lyapunov exponent of the system.
	"""
	set_seed(seed)
	g1 = Grid(**grid_params)
	g2 = copy.deepcopy(g1)

	if g2.agents:
		agent = g2.agents[0]
		nx, ny = agent.x + perturbation[0], agent.y + perturbation[1]
		if 0 <= nx < g2.height and 0 <= ny < g2.width and g2.is_empty(nx, ny):
			g2.remove_object(agent)
			agent.x, agent.y = nx, ny
			g2.place_object(agent)

	lyap, diffs = lyapunov_analysis(g1, g2, num_ticks, render, final_render)

	plt.figure(figsize=(8,5))
	plt.plot(range(len(diffs)), diffs, marker='o')
	plt.yscale('log')
	plt.title(f"Lyapunov Exponent Estimate: λ ≈ {lyap:.4f}")
	plt.xlabel("Tick")
	plt.ylabel("Normalized grid difference (log scale)")
	plt.grid(True)
	plt.show()

	return lyap


def check_determinism(num_ticks, seed, render=False, final_render=True, **grid_params):
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

	g1 = single_simulation(num_ticks, seed, render, final_render, **grid_params)
	g2 = single_simulation(num_ticks, seed, render, final_render, **grid_params)

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


def single_simulation(num_ticks=50, seed=123, render=False, final_render=True, **grid_params):
	"""
	Run and analyze a single grid simulation.

	The function initializes the grid, runs the simulation for the specified
	number of ticks, and prints the initial and final Shannon entropy values.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to execute. Default is 50.
	seed : int, optional
		Random seed ensuring reproducibility. Default is 123.

	Returns
	-------
	Grid
		The final grid state after simulation.
	"""
	set_seed(seed)
	grid = Grid(**grid_params)

	print("Initial Shannon entropy:", shannon_entropy(grid))

	grid.run_simulation(num_ticks, render)
	grid.render() if final_render == True else None

	print("Final Shannon entropy:", shannon_entropy(grid))
	return grid


def main_simulation(num_ticks=50, perturbation=(0, 1), seed=123, render=False, final_render=True, **grid_params):
	"""
	Execute the main experiment pipeline.

	This function:
	1. Runs a single grid simulation to observe entropy evolution.
	2. Runs a Lyapunov analysis to estimate system sensitivity to perturbations.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to run. Default is 50.
	perturbation : tuple of (int, int), optional
		Small positional offset applied to one agent for Lyapunov comparison.
	seed : int, optional
		Random seed to ensure deterministic behavior. Default is 123.

	Returns
	-------
	tuple of (Grid, float)
		Final grid state and estimated Lyapunov exponent.
	"""

	is_deterministic = check_determinism(num_ticks=num_ticks,seed=seed,render=render,final_render=final_render,**grid_params)

	if not is_deterministic:
		raise RuntimeError("Simulation environment is non-deterministic.")

	print("----- LYAPUNOV EXPONENT COMPARISON -----")
	lyap = compare_grids(num_ticks, perturbation, seed, render, final_render, **grid_params)
	print("Estimated Lyapunov exponent:", lyap)
	print("----- END OF LYAPUNOV EXPONENT COMPARISON -----")



# Main Execution
grid_params = {
	"width": 50,
	"height": 50,
	"metabolic_cost": 0.7,
	"min_child_energy": 6,
	"reproduction_cost": 4,
	"food_respawn_rate": 0.02,
	"num_agents": 20,
	"num_apples": 40,
	"num_oranges": 30,
	"num_walls": 60,
	"use_nn": False
}

main_simulation(
	num_ticks=1000,
	perturbation=(3, 3),
	seed=123,
	render=False,
	final_render=True, 
	**grid_params
)