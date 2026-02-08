import pandas as pd
import networkx as nx
import numpy as np
import math
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ==========================================
# 1. DATA LOADING & PARSING
# ==========================================

def load_sioux_falls_data(net_file, node_file):
    # Load Nodes
    nodes_df = pd.read_csv(node_file, sep='\t')
    # Clean up column names (strip whitespace)
    nodes_df.columns = [c.strip() for c in nodes_df.columns]

    # Load Links
    # The file format often has metadata at the top, we need to skip it
    # We'll try to read with pandas, skipping lines until we find the header
    with open(net_file, 'r') as f:
        lines = f.readlines()

    header_line = 0
    for i, line in enumerate(lines):
        if "FromNode" in line or "init_node" in line:
            header_line = i
            break

    links_df = pd.read_csv(net_file, skiprows=header_line, sep='\t')
    links_df.columns = [c.strip() for c in links_df.columns]

    return nodes_df, links_df


# ==========================================
# 2. DISASTER SIMULATION (Synthetic Data)
# ==========================================

def create_disaster_scenario(nodes_df, links_df, n_components=5):
    """
    Creates a synthetic disaster by clustering nodes into components
    and blocking all links that span between components.
    """
    # 1. Cluster nodes to create isolated regions
    coords = nodes_df[['X', 'Y']].values
    kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
    nodes_df['Component'] = kmeans.fit_predict(coords)

    # Create a mapping of NodeID -> Component
    node_comp_map = dict(zip(nodes_df['Node'], nodes_df['Component']))

    # 2. Build the Full Graph and Identify Blocked Links
    G = nx.Graph()
    blocked_links = []

    # Add nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row['Node'], pos=(row['X'], row['Y']), component=row['Component'])

    # Add edges
    for _, row in links_df.iterrows():
        u, v = int(row['init_node']), int(row['term_node'])
        # The paper uses 'FreeFlowTime' or similar as travel cost. 
        # Sioux Falls data usually has 'Free_Flow_Time' or 'Length'/'Capacity' -> BPR functions.
        # We will use 'Free_Flow_Time' if available, else 'Length'.
        cost = row.get('Free_Flow_Time', row.get('Length', 1))

        # Check if link crosses components
        comp_u = node_comp_map.get(u)
        comp_v = node_comp_map.get(v)

        if comp_u != comp_v:
            # This is a blocked link
            # Synthetic Repair Time: Random value between 5x and 20x the travel time
            repair_time = cost * random.uniform(5, 20)
            blocked_links.append({
                'u': u, 'v': v,
                'travel_time': cost,
                'repair_time': repair_time,
                'id': f"{min(u, v)}-{max(u, v)}"  # unique ID for undirected link
            })
            # We do NOT add this edge to G yet (it is broken)
        else:
            # Intact link
            G.add_edge(u, v, weight=cost)

    return G, nodes_df, blocked_links, node_comp_map


# ==========================================
# 3. GRAPH REDUCTION (Boundary Nodes & Complete Graph)
# ==========================================

def build_reduced_graph(G_intact, nodes_df, blocked_links, node_comp_map):
    """
    Builds the Complete Graph (CG) of boundary nodes.
    Calculates shortest paths (Travel Time TS, Repair Time TR) between all boundary nodes.
    """
    # 1. Identify Boundary Nodes
    # A boundary node is a node in a component that connects to a blocked link
    boundary_nodes = set()
    blocked_link_lookup = {}  # Map (u,v) -> repair_time

    for link in blocked_links:
        boundary_nodes.add(link['u'])
        boundary_nodes.add(link['v'])
        # Store for path checking
        edge = tuple(sorted((link['u'], link['v'])))
        blocked_link_lookup[edge] = link

    boundary_nodes = list(boundary_nodes)

    # 2. Calculate All-Pairs Shortest Paths (APSP) in the INTACT graph
    # Note: If components are fully isolated, path length is Infinity.
    # However, the "Complete Graph" logic in the paper implies we travel *between* components.
    # The paper says: "CG is a set of links among all nodes... whose lengths correspond to SP in undamaged G".
    # CRITICAL: We need the SP in the *Undamaged* graph (G_full), but we need to know 
    # which blocked links are traversed to sum up Repair Time (TR).

    G_full = G_intact.copy()
    for link in blocked_links:
        # Add blocked links back but track them
        G_full.add_edge(link['u'], link['v'], weight=link['travel_time'], is_blocked=True,
                        repair_cost=link['repair_time'])

    # Compute paths
    dist_matrix = {}  # Stores (TS, TR) tuple

    # We only need paths between boundary nodes
    # For a realistic size graph, we can run Dijkstra for each boundary node
    for start_node in boundary_nodes:
        lengths, paths = nx.single_source_dijkstra(G_full, start_node, weight='weight')

        for end_node in boundary_nodes:
            if start_node == end_node:
                dist_matrix[(start_node, end_node)] = (0, 0)
                continue

            if end_node not in paths:
                dist_matrix[(start_node, end_node)] = (float('inf'), float('inf'))
                continue

            # Reconstruct path to find Repair Cost (TR)
            path = paths[end_node]
            travel_time = lengths[end_node]
            repair_time_sum = 0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = G_full.get_edge_data(u, v)
                if edge_data.get('is_blocked', False):
                    repair_time_sum += edge_data['repair_cost']

            dist_matrix[(start_node, end_node)] = (travel_time, repair_time_sum)

    return boundary_nodes, dist_matrix, G_full


# ==========================================
# 4. ANT COLONY OPTIMIZATION (ACO)
# ==========================================

class ModifiedACO:
    def __init__(self, boundary_nodes, dist_matrix, node_comp_map, num_units, scenario=1,
                 alpha=1, beta=1, rho=0.2, num_ants=None, base_node=None):
        self.nodes = boundary_nodes
        self.dist_matrix = dist_matrix
        self.comp_map = node_comp_map
        self.m = num_units
        self.scenario = scenario  # 1 = No repair time cost, 2 = Repair time cost

        self.alpha = alpha
        self.beta = beta
        self.rho = rho  # Evaporation

        # Heuristic cache
        self.heuristics = {}
        self.pheromones = {}

        # Initialize Pheromones and Heuristics
        initial_pheromone = 0.5
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    ts, tr = self.dist_matrix.get((i, j), (float('inf'), 0))
                    if ts == float('inf') or ts == 0:
                        h = 0.0001
                    else:
                        h = 1.0 / ts  # Distance based heuristic (Scenario 1 & 2)

                    self.heuristics[(i, j)] = h
                    self.pheromones[(i, j)] = initial_pheromone

        # Base node: The paper assumes base is in one component. 
        # We pick the first boundary node's component as base.
        if base_node:
            self.base = base_node
        else:
            self.base = boundary_nodes[0]

        # Determine "Base Component"
        self.base_comp = self.comp_map[self.base]

    def _select_next_node(self, current_node, forbidden_comps, forbidden_nodes):
        """
        Standard ACO probabilistic selection.
        Returns selected node or None if no moves possible.
        """
        candidates = []
        weights = []

        for node in self.nodes:
            # We want to visit a NEW component.
            # So the target node must NOT be in a forbidden component.
            target_comp = self.comp_map[node]

            if target_comp not in forbidden_comps:
                # Valid move
                tau = self.pheromones.get((current_node, node), 0.1)
                eta = self.heuristics.get((current_node, node), 0.1)

                prob_numerator = (tau ** self.alpha) * (eta ** self.beta)
                candidates.append(node)
                weights.append(prob_numerator)

        if not candidates:
            return None

        # Roulette Wheel Selection
        total = sum(weights)
        if total == 0:
            return random.choice(candidates)

        probs = [w / total for w in weights]
        return random.choices(candidates, weights=probs, k=1)[0]

    def construct_solution(self):
        """
        Implements Algorithm 2.2 from the paper.
        Joint construction of routes for 'm' ants.
        """
        # State initialization
        ant_positions = [self.base] * self.m
        ant_times = [0.0] * self.m  # Time elapsed for each ant
        ant_routes = [[self.base] for _ in range(self.m)]

        # Track visited components (Goal: Visit all)
        # Base component is already "visited"
        visited_comps = {self.base_comp}

        # Global components list to check termination
        all_comps = set(self.comp_map.values())

        # Loop until all components visited
        while visited_comps != all_comps:

            # 1. Evaluate potential moves for ALL ants
            best_choice = None
            best_char_val = -1

            # The paper says: "GenerateNewNode" using prob, then calculate "char[a]".
            # Then pick ONE ant to move based on char[a].

            # Stores (ant_idx, target_node) candidates
            candidates = []
            char_weights = []

            for ant_idx in range(self.m):
                curr = ant_positions[ant_idx]

                # Get a target suggestion for this ant
                target = self._select_next_node(curr, visited_comps, None)

                if target is not None:
                    # Calculate 'char' value (Eq 13)
                    # char = 1 / (S_ai + t_ij + rt_ij)
                    # S_ai = ant_times[ant_idx] (Time accumulated so far)
                    # t_ij = travel time
                    # rt_ij = repair time (depends on scenario)

                    ts, tr = self.dist_matrix.get((curr, target))

                    repair_cost = 0
                    if self.scenario == 2:
                        repair_cost = tr

                    # Note: The paper minimizes TIME. High char = Low Time.
                    denominator = ant_times[ant_idx] + ts + repair_cost
                    if denominator == 0: denominator = 0.001

                    char_val = 1.0 / denominator

                    candidates.append((ant_idx, target, ts, repair_cost))
                    char_weights.append(char_val)

            if not candidates:
                # No ant can move to a new component? 
                # This happens if graph is disconnected or logic error.
                # Just break to avoid infinite loop.
                break

            # 2. Select ONE ant to move
            # Probabilistic selection based on 'char'
            total_char = sum(char_weights)
            probs = [c / total_char for c in char_weights]

            choice_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
            winner_ant, target_node, cost_ts, cost_tr = candidates[choice_idx]

            # 3. Update State
            # Move the ant
            ant_positions[winner_ant] = target_node
            ant_routes[winner_ant].append(target_node)

            # Update time
            # Scenario 1: Add TS only. Scenario 2: Add TS + TR.
            if self.scenario == 1:
                ant_times[winner_ant] += cost_ts
            else:
                ant_times[winner_ant] += (cost_ts + cost_tr)

            # Mark component as visited
            new_comp = self.comp_map[target_node]
            visited_comps.add(new_comp)

        # 4. Return to base (optional, but paper implies closed tours usually)
        # The paper says: "We further require the repair units to return to their starting location"
        for i in range(self.m):
            curr = ant_positions[i]
            ts, tr = self.dist_matrix.get((curr, self.base), (0, 0))
            ant_routes[i].append(self.base)
            if self.scenario == 1:
                ant_times[i] += ts
            else:
                ant_times[i] += (ts + tr)  # Assuming paying repair on way back?
                # Usually way back is cleared path, but let's stick to consistent logic.
                # Actually, if path was cleared by *any* ant, cost is 0.
                # This complexity is handled simply here by assuming worst case or
                # trusting the "Scenario 1" abstraction.

        return ant_routes, ant_times

    def optimize(self, iterations=50):
        best_routes = None
        best_times = [float('inf')] * self.m
        best_objective_val = (float('inf'),)  # Lexicographical tuple

        print(f"Starting ACO optimization (Scenario {self.scenario})...")

        for it in range(iterations):
            # 1. Construct Solutions
            # Paper uses 'numb_of_colonies' or multiple ants per iter. 
            # We'll generate N solutions per iteration and pick best.
            iter_routes = []
            iter_times = []

            for _ in range(10):  # Colony size 10
                routes, times = self.construct_solution()
                iter_routes.append(routes)
                iter_times.append(times)

            # 2. Find Best in Iteration (Lexicographical)
            # Sort times descending for comparison
            # Objective: Minimize [Max(T), 2ndMax(T)...]

            current_best_idx = -1
            current_best_sort = [float('inf')] * self.m

            for idx, t_vec in enumerate(iter_times):
                sorted_t = sorted(t_vec, reverse=True)
                if sorted_t < current_best_sort:
                    current_best_sort = sorted_t
                    current_best_idx = idx

            # 3. Update Global Best
            if current_best_sort < sorted(best_times, reverse=True):
                best_times = iter_times[current_best_idx]
                best_routes = iter_routes[current_best_idx]

            # 4. Pheromone Update (MMAS)
            # Evaporation
            for k in self.pheromones:
                self.pheromones[k] *= (1.0 - self.rho)
                if self.pheromones[k] < 0.01: self.pheromones[k] = 0.01  # Min pheromone

            # Deposit on BEST route
            # Amount: 1 / Total Time (or similar)
            deposit = 100.0 / (sum(best_times) + 1)

            for r_idx, route in enumerate(best_routes):
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    # Update both directions for undirected graph logic
                    if (u, v) in self.pheromones:
                        self.pheromones[(u, v)] += deposit
                    if (v, u) in self.pheromones:
                        self.pheromones[(v, u)] += deposit

            # Max pheromone clamp
            for k in self.pheromones:
                if self.pheromones[k] > 5.0: self.pheromones[k] = 5.0

        return best_routes, best_times


# ==========================================
# MAIN EXECUTION
# ==========================================

# File names based on user upload
node_file = "SiouxFalls_node.tntp.txt"
net_file = "SiouxFalls_net.tntp.txt"

# 1. Parse
nodes_df, links_df = load_sioux_falls_data(net_file, node_file)

# 2. Create Disaster
G_synthetic, nodes_info, blocked_links, comp_map = create_disaster_scenario(nodes_df, links_df)

# 3. Reduction
print("Building Reduced Graph...")
boundary_nodes, dist_matrix, G_full = build_reduced_graph(G_synthetic, nodes_info, blocked_links, comp_map)
print(f"Graph Reduced: {len(nodes_info)} original nodes -> {len(boundary_nodes)} boundary nodes.")
print(f"Components created: {len(set(comp_map.values()))}")
print(f"Blocked Links: {len(blocked_links)}")

# 4. Run ACO
# Number of repair units
NUM_REPAIR_UNITS = 4

# Scenario 1
print("\n=== RUNNING SCENARIO 1 (Minimize Travel Time, No Repair Cost) ===")
aco_s1 = ModifiedACO(boundary_nodes, dist_matrix, comp_map, num_units=NUM_REPAIR_UNITS, scenario=1)
routes_s1, times_s1 = aco_s1.optimize(iterations=50)

print("\nResults Scenario 1:")
for i, (r, t) in enumerate(zip(routes_s1, times_s1)):
    print(f"Unit {i + 1}: Time {t:.2f}, Route: {r}")
print(f"Sorted Times: {sorted(times_s1, reverse=True)}")

# Scenario 2
print("\n=== RUNNING SCENARIO 2 (Minimize Travel + Repair Time) ===")
aco_s2 = ModifiedACO(boundary_nodes, dist_matrix, comp_map, num_units=NUM_REPAIR_UNITS, scenario=2)
routes_s2, times_s2 = aco_s2.optimize(iterations=50)

print("\nResults Scenario 2:")
for i, (r, t) in enumerate(zip(routes_s2, times_s2)):
    print(f"Unit {i + 1}: Time {t:.2f}, Route: {r}")
print(f"Sorted Times: {sorted(times_s2, reverse=True)}")