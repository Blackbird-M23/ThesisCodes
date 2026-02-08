import pandas as pd
import networkx as nx
import random


# ==========================================
# 1. DATA LOADING
# ==========================================

def load_sioux_falls_data(net_file, node_file):
    # Load Nodes
    nodes_df = pd.read_csv(node_file, sep='\t')
    nodes_df.columns = [c.strip() for c in nodes_df.columns]

    # Load Links
    # Skip metadata lines until header found
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


def load_disaster_input(csv_file):
    """
    Reads the user-defined blocked links and repair times.
    Expected columns: NodeA, NodeB, RepairTime
    """
    try:
        df = pd.read_csv(csv_file)
        # Ensure correct column names/types
        expected_cols = {'NodeA', 'NodeB', 'RepairTime'}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {expected_cols}")
        return df
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please create it first.")
        exit()


# ==========================================
# 2. DISASTER MODELING (From Input File)
# ==========================================

def apply_disaster_scenario(nodes_df, links_df, blocked_df):
    """
    1. Builds the full graph.
    2. Identifies blocked links from input CSV.
    3. Removes them to find 'Components'.
    4. Returns graph with component info and blocked link details.
    """
    G = nx.Graph()

    # 1. Build Full Graph (Initially Intact)
    # We create a lookup for normal travel times first
    link_travel_times = {}

    for _, row in links_df.iterrows():
        u, v = int(row['init_node']), int(row['term_node'])
        cost = row.get('Free_Flow_Time', row.get('Length', 1))

        # Store travel time (min of both directions if directed in file, but we treat as undirected for connectivity)
        edge_key = tuple(sorted((u, v)))
        if edge_key not in link_travel_times:
            link_travel_times[edge_key] = cost

        # Add to graph
        G.add_edge(u, v, weight=cost)

    # 2. Process Blocked Links
    # We need to map the user's blocked links to the graph edges
    #blocked_edges = []

    # Helper to check if edge exists in CSV
    # We use a set for faster lookup of (u, v) pairs
    user_blocks = set()
    block_info = {}  # Store repair times, used DICTIONARY

    for _, row in blocked_df.iterrows():
        u, v = int(row['NodeA']), int(row['NodeB'])
        repair_time = float(row['RepairTime'])

        key = tuple(sorted((u, v)))
        user_blocks.add(key) #user_defined_blocks
        block_info[key] = repair_time

    # 3. Remove Blocked Links to define Components
    # We iterate over the graph's edges and remove those in the block list
    G_broken = G.copy()
    actual_blocked_links = []  #LIST

    for u, v in list(G.edges()):
        key = tuple(sorted((u, v)))
        if key in user_blocks:
            G_broken.remove_edge(u, v)

            # Retrieve travel time (TS) from original data
            t_travel_time = link_travel_times.get(key, 1)
            # Repair time from blocked_links_generated
            t_repair_time = block_info[key]

            actual_blocked_links.append({
                'u': u, 'v': v,
                'travel_time': t_travel_time,
                'repair_time': t_repair_time
            })

    # 4. Identify Components (Islands)
    components = list(nx.connected_components(G_broken))
    node_comp_map = {} # set of node component for mapping
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            node_comp_map[node] = comp_id # node component map holds the component ID of the node that it belongs to

    print(f"Disaster Scenario Applied: Graph broken into {len(components)} components.")
    # print components/islands
    for i, comp in enumerate(components):
        print(f"Component {i}: {sorted(list(comp))}")

    return G, actual_blocked_links, node_comp_map


# ==========================================
# 3. GRAPH REDUCTION
# ==========================================

def build_reduced_graph(G_full, blocked_links, node_comp_map):
    """
    Builds the reduced graph as defined in the paper.
    Nodes = boundary nodes
    Edges (i,j) only if:
      - i and j are in DIFFERENT components
      - The shortest path between them crosses at least ONE damaged link
    Edge weight = (TS_ij, TR_ij)
    """

    # ==========================================
    # 1. Identify Boundary Nodes
    # ==========================================
    boundary_nodes = set()

    for link in blocked_links:
        u, v = link['u'], link['v']
        comp_u = node_comp_map.get(u)
        comp_v = node_comp_map.get(v)

        # Only if this damaged link connects DIFFERENT components
        if comp_u is not None and comp_v is not None and comp_u != comp_v:
            boundary_nodes.add(u)
            boundary_nodes.add(v)

    boundary_nodes = list(boundary_nodes)
    print("Set of Boundary Nodes:", end=" ")
    for node in boundary_nodes:
        print(node, end=" ")

    print(f"\nBoundary nodes identified: {len(boundary_nodes)}")

    # ==========================================
    # 2. Build traversal graph (intact network)
    # ==========================================
    G_traversal = G_full.copy()

    # Lookup table for damaged links and their repair times
    blocked_lookup = {
        tuple(sorted((x['u'], x['v']))): x['repair_time']
        for x in blocked_links
    }

    # ==========================================
    # 3. Build Reduced Graph Distance Matrix
    # ==========================================
    dist_matrix = {}

    for start_node in boundary_nodes:

        # Shortest paths based on TRAVEL TIME ONLY (as in the paper)
        lengths, paths = nx.single_source_dijkstra(
            G_traversal, start_node, weight='weight'
        )

        for end_node in boundary_nodes:

            if start_node == end_node:
                continue

            # --------- PAPER CONSTRAINT 1 ----------
            # Only consider pairs from DIFFERENT COMPONENTS
            if node_comp_map[start_node] == node_comp_map[end_node]:
                continue

            if end_node not in paths:
                continue

            path = paths[end_node]

            # Compute TS and TR along this shortest path
            total_ts = 0.0
            total_tr = 0.0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = G_traversal.get_edge_data(u, v)

                # Add travel time
                total_ts += edge_data.get('weight', 1)

                # Add repair time IF this edge is damaged
                edge_key = tuple(sorted((u, v)))
                if edge_key in blocked_lookup:
                    total_tr += blocked_lookup[edge_key]

            # --------- PAPER CONSTRAINT 2 ----------
            # Path MUST include at least ONE damaged link
            # Otherwise this edge does not represent a repair action
            if total_tr > 0:
                dist_matrix[(start_node, end_node)] = (total_ts, total_tr)

    print(f"Reduced graph edges created: {len(dist_matrix)}")

    return boundary_nodes, dist_matrix


# def build_reduced_graph(G_full, blocked_links_info, node_comp_map):
#     """
#     Builds the complete graph of boundary nodes.
#     Calculates TS (Travel Time) and TR (Repair Cost) using shortest paths.
#     """
#     # 1. Identify Boundary Nodes
#     boundary_nodes = set()
#
#     # Identify which links are actually bridges between components
#     # (Some blocked links might be inside a component if redundancy exists,
#     # but strictly speaking, the paper focuses on links connecting components)
#
#     for link in blocked_links_info:
#         u, v = link['u'], link['v']
#         comp_u = node_comp_map.get(u)
#         comp_v = node_comp_map.get(v)
#
#         # Only treat as boundary if they connect DIFFERENT components
#         if comp_u is not None and comp_v is not None and comp_u != comp_v:  # checks if u and v belongs to a component and the component they belong to are different
#             boundary_nodes.add(u)
#             boundary_nodes.add(v)
#
#     boundary_nodes = list(boundary_nodes)
#
#     # 2. Build Distance Matrix (TS and TR)
#     # We calculate shortest path on the FULL graph (including blocked links with penalties)
#     # to find the optimal path between boundaries.
#
#     # Construct a traversal graph where blocked links exist but have properties
#     G_traversal = G_full.copy()
#
#     # Mark blocked edges in G_traversal
#     blocked_lookup = {tuple(sorted((x['u'], x['v']))): x['repair_time'] for x in blocked_links_info}
#     # This creates : (3,7) → repair_time, (5,9) → repair_time
#
#     # Nodes = boundary nodes
#     # Edge (i,j) exists if a path exists between boundary nodes i and j in the original graph
#     # Each reduced edge has:
#     # TSᵢⱼ = sum of travel times along shortest path
#     # TRᵢⱼ = sum of repair times of damaged links on that path
#
#     dist_matrix = {}
#
#     for start_node in boundary_nodes:
#         # We compute Dijkstra weighted by TRAVEL TIME only initially to find shortest geometric path
#         # (The paper implies shortest physical path is used, then costs are summed)
#         # Shortest path is determined by travel time only; repair times are accumulated afterwards.
#         lengths, paths = nx.single_source_dijkstra(G_traversal, start_node, weight='weight')
#
#         for end_node in boundary_nodes:
#             if start_node == end_node:
#                 dist_matrix[(start_node, end_node)] = (0, 0)
#                 continue
#
#             if end_node not in paths:
#                 continue
#
#             path = paths[end_node]
#
#             # Calculate TS (Travel Time) and TR (Repair Time) for this path
#             total_ts = 0
#             total_tr = 0
#
#             for i in range(len(path) - 1):
#                 u, v = path[i], path[i + 1]
#                 edge_data = G_traversal.get_edge_data(u, v)
#
#                 # TS
#                 total_ts += edge_data.get('weight', 1)
#
#                 # TR - Check if this specific link is blocked
#                 edge_key = tuple(sorted((u, v)))
#                 if edge_key in blocked_lookup:
#                     total_tr += blocked_lookup[edge_key]
#
#             dist_matrix[(start_node, end_node)] = (total_ts, total_tr)
#
#     return boundary_nodes, dist_matrix


# ==========================================
# 4. ACO ALGORITHM
# ==========================================

class ModifiedACO:
    def __init__(self, boundary_nodes, dist_matrix, node_comp_map, num_units, base_node, G_full, scenario=1):
        self.nodes = boundary_nodes
        self.dist_matrix = dist_matrix
        self.comp_map = node_comp_map
        self.m = num_units
        self.scenario = scenario
        self.G_full = G_full

        # ACO Parameters
        self.alpha = 1.0
        self.beta = 1.0
        self.rho = 0.2

        # State
        self.heuristics = {}
        self.pheromones = {}

        # -------------------------------
        # USER-DEFINED BASE STATION
        # -------------------------------
        if base_node not in self.comp_map:
            raise ValueError(f"Base node {base_node} does not exist in the network.")

        self.base = base_node
        self.artificial_base_comp_id = -1
        
        # Pre-calculate Base -> Boundary Distances (Travel Time only)
        # Using intact graph (G_full)
        lengths, _ = nx.single_source_dijkstra(self.G_full, self.base, weight='weight')
        self.base_dists = {}
        for bn in self.nodes:
            self.base_dists[bn] = lengths.get(bn, float('inf'))

        print(f"Base station set to node {self.base} (Artificial Component {self.artificial_base_comp_id})")

        # Init Pheromones/Heuristics
        # 1. Boundary <-> Boundary
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    ts, _ = self.dist_matrix.get((i, j), (float('inf'), 0))
                    # Heuristic 1/Time (avoid div by zero)
                    h = 1.0 / (ts + 0.001) if ts < float('inf') else 0.0
                    self.heuristics[(i, j)] = h
                    self.pheromones[(i, j)] = 0.5  # Init pheromone
        
        # 2. Base -> Boundary (One-way for selection)
        for bn in self.nodes:
            ts = self.base_dists[bn]
            h = 1.0 / (ts + 0.001) if ts < float('inf') else 0.0
            self.heuristics[(self.base, bn)] = h
            self.pheromones[(self.base, bn)] = 0.5


    def _select_next(self, curr, visited_comps):
        # Build probabilities
        candidates = []
        weights = []

        # CASE 1: Start at Base
        # if curr == self.base:
        #     targets = self.nodes
        # else:
        #     # CASE 2: Boundary to Boundary
        #     targets = self.nodes
        targets = self.nodes

        for node in targets:
            if node == curr: continue
##           # if node == self.base:
##           #     continue

            target_comp = self.comp_map[node]
            if target_comp not in visited_comps:
                
                # Fetch Pheromone & Heuristic
                if curr == self.base:
                     ph = self.pheromones.get((self.base, node), 0.1)
                     he = self.heuristics.get((self.base, node), 0.0001)
                else:
                     # Validate connectivity in reduced graph
                     if (curr, node) not in self.dist_matrix:
                         continue
                     ph = self.pheromones.get((curr, node), 0.1)
                     he = self.heuristics.get((curr, node), 0.0001)
                
                weight = (ph ** self.alpha) * (he ** self.beta)

                candidates.append(node)
                weights.append(weight)

        if not candidates: return None

        # Roulette
        total = sum(weights)
        if total == 0: return random.choice(candidates)
        probs = [w / total for w in weights]
        return random.choices(candidates, weights=probs, k=1)[0]  # through choosing between the probs we are choosing a candidates

    def construct_solution(self):
        # Each ant starts at base
        pos = [self.base] * self.m  #All ants start from base , just a normal LIST
        times = [0.0] * self.m  # Time spent by each ant
        routes = [[self.base] for _ in range(self.m)]  # A LIST of LISTS

        visited_comps = {self.artificial_base_comp_id}
        # We need to visit all REAL components
        all_real_comps = set(self.comp_map.values())

        # Step-by-step construction
        # Stop when we have visited all real components + the artificial base
        while len(visited_comps) < len(all_real_comps) + 1:

            # Identify valid moves for ALL ants
            moves = [] #This will store all valid moves ants can make at this step.

            for ant_idx in range(self.m):
                curr_node = pos[ant_idx]
                target = self._select_next(curr_node, visited_comps)

                if target is not None:
                    # Determine cost
                    if curr_node == self.base:
                        # Base -> Boundary (Travel Time only, No Repair)
                        ts = self.base_dists[target]
                        tr = 0.0
                    else:
                        # Boundary -> Boundary
                        # Explicit lookup, no fallbacks
                        if (curr_node, target) in self.dist_matrix:
                            ts, tr = self.dist_matrix[(curr_node, target)]
                        else:
                            # Should not happen given _select_next checks, but careful
                            continue

                    # Cost calculation based on Scenario
                    repair_cost = tr if self.scenario == 2 else 0

                    # Eq 13: Attractiveness char
                    # Note: We add a small epsilon to avoid div by zero if times are 0
                    denom = times[ant_idx] + ts + repair_cost + 0.001
                    char_val = 1.0 / denom

                    moves.append({
                        'ant': ant_idx,
                        'target': target,
                        'ts': ts,
                        'tr': tr,
                        'char': char_val
                    })

            if not moves: break # If no ant cant find the next node to go to -> moves list will be empty after all the iteration → stop

            # Probabilistic selection of WHICH ANT moves
            total_char = sum(m['char'] for m in moves)
            probs = [m['char'] / total_char for m in moves]

            choice = random.choices(moves, weights=probs, k=1)[0]  # through choosing between the probs we are choosing a move
            # only one ant move each iteration NOT all the ants

            # Apply Move
            ant = choice['ant']
            target = choice['target']

            pos[ant] = target  # current position of ant after the move
            routes[ant].append(target) # adding the new position of that chosen ant

            # Update Time
            cost = choice['ts'] + (choice['tr'] if self.scenario == 2 else 0)
            times[ant] += cost

            # Update Visited
            visited_comps.add(self.comp_map[target])

        # Return to base logic (FORCED)
        for i in range(self.m):
            curr = pos[i]
            
            if curr == self.base:
                ts = 0.0
                tr = 0.0
            else:
                # Return cost: Base Dist from current node + 0 repair
                ts = self.base_dists[curr]
                tr = 0.0
            
            routes[i].append(self.base)
            cost = ts + (tr if self.scenario == 2 else 0)
            times[i] += cost

        return routes, times

    def run(self, iterations=100):
        best_routes = None
        # Init best times with infinity
        best_times = [float('inf')] * self.m
        best_times_sorted = [float('inf')] * self.m

        for _ in range(iterations):
            # Run colony (10 ants/solutions per iter)
            iter_solutions = [self.construct_solution() for _ in range(10)]

            # Find best in this batch (Lexicographical Min)
            # Sort times desc: [max_time, 2nd_max, ...]
            # Python compares lists element by element, which matches lexicographical rank
            batch_best_sol = min(iter_solutions, key=lambda x: sorted(x[1], reverse=True))
            batch_routes, batch_times = batch_best_sol

            # Compare with global best
            if sorted(batch_times, reverse=True) < best_times_sorted:
                best_times_sorted = sorted(batch_times, reverse=True)
                best_routes = batch_routes
                best_times = batch_times

            # Pheromone Update (Simplistic MMAS)
            # Evaporate
            for k in self.pheromones:
                self.pheromones[k] *= (1.0 - self.rho)

            # Deposit on Batch Best
            reward = 100.0 / (sum(batch_times) + 1.0)
            for r in batch_routes:
                for i in range(len(r) - 1):

                    key = (r[i], r[i + 1])
                    if key in self.pheromones:
                        self.pheromones[key] += reward
##                    # if r[i] != self.base and r[i + 1] != self.base:
##                    #     self.pheromones[(r[i], r[i + 1])] += reward

                    # Undirected pheromone update
                    key_rev = (r[i + 1], r[i])
                    if key_rev in self.pheromones:
                        self.pheromones[key_rev] += reward

        return best_routes, best_times


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # File Paths
    NODE_FILE = "SiouxFalls_node.tntp.txt"
    NET_FILE = "SiouxFalls_net.tntp.txt"
    DISASTER_FILE = "blocked_links_generated.csv"

    # 1. Load Data
    print("Loading Data...")
    nodes_df, links_df = load_sioux_falls_data(NET_FILE, NODE_FILE)
    blocked_df = load_disaster_input(DISASTER_FILE)

    # 2. Apply Disaster
    G_full, blocked_links_info, comp_map = apply_disaster_scenario(nodes_df, links_df, blocked_df)

    # 3. Reduce Graph
    print("Reducing Graph...")
    boundary_nodes, dist_matrix = build_reduced_graph(G_full, blocked_links_info, comp_map)
    print(f"Boundary Nodes: {len(boundary_nodes)}")

    # ==========================================
    # USER INPUT: BASE STATION NODE
    # ==========================================

    print("\nAvailable nodes in the network:", sorted(comp_map.keys()))

    while True:
        try:
            base_node = int(input("\nEnter base station node number: "))

            if base_node not in comp_map:
                print("❌ Invalid node number. Try again.")
                continue

            # Recommend base to be in the largest (main) component
            base_comp = comp_map[base_node]
            print(f"✔ Base node {base_node} belongs to Component {base_comp}")

            break

        except ValueError:
            print("❌ Please enter a valid integer node number.")

    # node_to_find = 15
    # found = False
    #
    # # Iterate through every single (start_node, end_node) tuple key in the dictionary
    # for start_node, end_node in dist_matrix.keys():
    #     if start_node == node_to_find or end_node == node_to_find:
    #         found = True
    #         break  # Stop searching as soon as we find it
    #
    # if found:
    #     print("YES")
    # else:
    #     print("NO")

    # 4. Execute Scenarios
    NUM_UNITS = 4

    print("\n--- Running Scenario 1 (No Repair Cost) ---")
    aco1 = ModifiedACO(boundary_nodes, dist_matrix, comp_map, NUM_UNITS, base_node=base_node, G_full=G_full, scenario=1)
    routes1, times1 = aco1.run(iterations=100)

    print("\nResults Scenario 1:")
    for i, (r, t) in enumerate(zip(routes1, times1)):
        # Calculate repair cost
        repair_cost = 0
        for idx in range(len(r) - 1):
            u, v = r[idx], r[idx+1]
            if (u, v) in dist_matrix:
                _, tr = dist_matrix[(u, v)]
            else:
                tr = 0
            repair_cost += tr
        print(f"Unit {i + 1}: Time {t:.2f}, Repair Cost {repair_cost:.2f}, Route: {r}")
    print(f"Sorted Times: {sorted(times1, reverse=True)}")

    print("\n--- Running Scenario 2 (With Repair Cost) ---")
    aco2 = ModifiedACO(boundary_nodes, dist_matrix, comp_map, NUM_UNITS, base_node=base_node, G_full=G_full, scenario=2)
    routes2, times2 = aco2.run(iterations=200)

    print("\nResults Scenario 2:")
    for i, (r, t) in enumerate(zip(routes2, times2)):
        # Calculate repair cost
        repair_cost = 0
        for idx in range(len(r) - 1):
            u, v = r[idx], r[idx+1]
            if (u, v) in dist_matrix:
                _, tr = dist_matrix[(u, v)]
            else:
                tr = 0
            repair_cost += tr
        print(f"Unit {i + 1}: Time {t:.2f}, Repair Cost {repair_cost:.2f}, Route: {r}")
    print(f"Sorted Times: {sorted(times2, reverse=True)}")


