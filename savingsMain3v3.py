import pandas as pd
import networkx as nx
import random


# .\.venv\Scripts\Activate.ps1  : activate

# ==========================================
# 1. DATA LOADING
# ==========================================

def load_sioux_falls_data(net_file, node_file):
    # Load Nodes (using \s+ handles spaces or tabs dynamically)
    nodes_df = pd.read_csv(node_file, sep=r'\s+')
    nodes_df.columns = [c.strip().lower() for c in nodes_df.columns]

    # Load Links directly from the new CSV file
    links_df = pd.read_csv(net_file)
    # Ensure all columns are lowercase and stripped of extra spaces
    links_df.columns = [c.strip().lower() for c in links_df.columns]

    # ==========================================
    # VERIFICATION: Print the parsed data
    # ==========================================
    # print("\n--- Verifying Loaded Nodes Data ---")
    # print(f"Total Nodes: {len(nodes_df)}")
    # print(nodes_df.to_string()) 15, 10, 9, 5, 6, 8, 9, 10, 15

    print("\n--- Verifying Loaded Network (Links) Data ---")
    print("Columns found:", links_df.columns.tolist())
    print(f"Total Links: {len(links_df)}")
    cols_to_print = [c for c in ['init_node', 'term_node', 'capacity', 'length', 'free_flow_time'] if c in links_df.columns]
    print(links_df[cols_to_print].to_string())
    print("-------------------------------------\n")

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
        cost = row.get('free_flow_time', row.get('length', 1))

        # Store travel time (min of both directions if directed in file, but we treat as undirected for connectivity)
        edge_key = tuple(sorted((u, v)))
        if edge_key not in link_travel_times:
            link_travel_times[edge_key] = cost

        # Add to graph
        G.add_edge(u, v, weight=cost)

    # 2. Process Blocked Links
    # We need to map the user's blocked links to the graph edges
    # blocked_edges = []

    # Helper to check if edge exists in CSV
    # We use a set for faster lookup of (u, v) pairs
    user_blocks = set()
    block_info = {}  # Store repair times, used DICTIONARY

    for _, row in blocked_df.iterrows():
        u, v = int(row['NodeA']), int(row['NodeB'])
        repair_time = float(row['RepairTime'])

        key = tuple(sorted((u, v)))
        user_blocks.add(key)

        # FIX: keep MAX repair time instead of overwriting
        if key in block_info:
            block_info[key] = max(block_info[key], repair_time)
        else:
            block_info[key] = repair_time

    # 3. Remove Blocked Links to define Components
    # We iterate over the graph's edges and remove those in the block list
    G_broken = G.copy()
    actual_blocked_links = []  # LIST

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
    node_comp_map = {}  # set of node component for mapping
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            node_comp_map[node] = comp_id  # node component map holds the component ID of the node that it belongs to

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
        if comp_u is not None and comp_v is not None:
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
        print(f"\n--- Results for Start Node: {start_node} ---")

        # 1. Clean up lengths (Distances)
        # We want the values, not the keys
        clean_lengths = {node: float(dist) for node, dist in lengths.items()}
        print(f"DISTANCES: {clean_lengths}")

        # 2. Clean up paths
        # Paths are lists of integers (Node IDs). Usually, we keep these as ints.
        # If you want them as floats for some reason, you have to map them inside the list:
        clean_paths = {node: [float(n) for n in path_list] for node, path_list in paths.items()}
        print(f"PATHS: {clean_paths}")

        for end_node in boundary_nodes:
            if start_node == end_node:
                continue

            # --------- PAPER CONSTRAINT 1 ----------
            # Only consider pairs from DIFFERENT COMPONENTS
            # if node_comp_map[start_node] == node_comp_map[end_node]:
            #     continue

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


def get_full_path(G_full, route):
    """
    Expands a route of boundary nodes into a full path through the intact network.
    """
    if not route:
        return []

    full_path = [route[0]]
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        try:
            # Find the shortest path between consecutive nodes in the route
            path = nx.shortest_path(G_full, source=u, target=v, weight='weight')
            # Extend full_path but skip the first node to avoid duplicates (e.g., A->B, B->C)
            full_path.extend(path[1:])
        except nx.NetworkXNoPath:
            # Fallback if no path exists (though there should be one in the full graph)
            full_path.append(v)

    return full_path

# ==========================================
# NEW: Consistent TS/TR computation from FULL PATH
# ==========================================
def compute_path_TS_TR_with_repair(path, G, blocked_lookup, repaired_edges):
    """
    Computes TS and TR from actual path.
    Also TRACKS repaired edges to avoid double counting.

    UPDATED:
    - Ensures consistency between ACO and real execution
    - Prevents double repair cost
    """

    total_ts = 0.0
    total_tr = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v)

        total_ts += edge_data.get('weight', 1)

        edge_key = tuple(sorted((u, v)))

        # Only charge repair if NOT already repaired
        if edge_key in blocked_lookup and edge_key not in repaired_edges:
            total_tr += blocked_lookup[edge_key]
            repaired_edges.add(edge_key)  # ✅ mark as repaired

    return total_ts, total_tr

class ModifiedACO:
    def __init__(self, boundary_nodes, dist_matrix, node_comp_map,
                 num_units, base_node, G_full, scenario=1):

        self.nodes = boundary_nodes
        self.dist_matrix = dist_matrix
        self.comp_map = node_comp_map
        self.m = num_units
        self.scenario = scenario
        self.G_full = G_full

        # ACO parameters
        self.alpha = 1.0
        self.beta = 1.0
        self.rho = 0.2

        self.heuristics = {}
        self.pheromones = {}

        if base_node not in self.comp_map:
            raise ValueError("Base node not in network")

        self.base = base_node
        self.artificial_base_comp_id = -1

        # -------------------------------
        # Base → Boundary distances
        # -------------------------------
        lengths, _ = nx.single_source_dijkstra(self.G_full, self.base, weight='weight')
        self.base_dists = {bn: lengths.get(bn, float('inf')) for bn in self.nodes}

        # -------------------------------
        # Heuristic Initialization
        # -------------------------------
        for i in self.nodes:
            for j in self.nodes:
                if i != j and (i, j) in self.dist_matrix:
                    ts, _ = self.dist_matrix[(i, j)]
                    self.heuristics[(i, j)] = 1.0 / (ts + 0.001)

        for bn in self.nodes:
            ts = self.base_dists[bn]
            self.heuristics[(self.base, bn)] = 1.0 / (ts + 0.001)

        # -------------------------------
        # Savings Initialization
        # -------------------------------
        init_routes, init_times = self.savings_init()

        self.best_routes = init_routes
        self.best_times = init_times
        self.best_times_sorted = sorted(init_times, reverse=True)

        # -------------------------------
        # MMAS Pheromone Initialization (Eq 12)
        # -------------------------------
        lengthpath = sum(init_times)
        tau0 = 1.0 / (lengthpath + 1e-6)

        TAU_MIN = 0.01
        TAU_MAX = 1.0
        tau0 = max(TAU_MIN, min(TAU_MAX, tau0))

        # Initialize ALL edges uniformly
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    self.pheromones[(i, j)] = tau0

        for bn in self.nodes:
            self.pheromones[(self.base, bn)] = tau0

    # ==========================================================
    # SAVINGS ALGORITHM (Robust Version from Implementation A)
    # ==========================================================
    def savings_init(self):

        routes = {node: [self.base, node, self.base] for node in self.nodes}
        node_route_map = {node: node for node in self.nodes}

        savings = []

        for i in self.nodes:
            for j in self.nodes:
                if i == j:
                    continue
                if (i, j) not in self.dist_matrix:
                    continue

                d0i = self.base_dists[i]
                d0j = self.base_dists[j]
                dij = self.dist_matrix[(i, j)][0]

                Sij = d0i + d0j - dij
                savings.append((Sij, i, j))

        savings.sort(reverse=True, key=lambda x: x[0])

        for Sij, i, j in savings:

            ri_id = node_route_map[i]
            rj_id = node_route_map[j]

            if ri_id == rj_id:
                continue

            ri = routes[ri_id]
            rj = routes[rj_id]

            if ri[-2] == i and rj[1] == j:
                merged = ri[:-1] + rj[1:]

                routes[ri_id] = merged
                del routes[rj_id]

                for node in merged:
                    if node != self.base:
                        node_route_map[node] = ri_id

            if len(routes) <= self.m:
                break

        final_routes = list(routes.values())

        while len(final_routes) > self.m:
            r1 = final_routes.pop()
            r2 = final_routes.pop()
            merged = r1[:-1] + r2[1:]
            final_routes.append(merged)

        route_times = []
        for route in final_routes:
            total = 0.0
            for k in range(len(route) - 1):
                u, v = route[k], route[k + 1]

                if u == self.base:
                    total += self.base_dists[v]
                elif v == self.base:
                    total += self.base_dists[u]
                else:
                    total += self.dist_matrix[(u, v)][0]

            route_times.append(total)

            # ... (existing code inside savings_init)
            route_times = []
            for route in final_routes:
                total = 0.0
                for k in range(len(route) - 1):
                    u, v = route[k], route[k + 1]

                    if u == self.base:
                        total += self.base_dists[v]
                    elif v == self.base:
                        total += self.base_dists[u]
                    else:
                        total += self.dist_matrix[(u, v)][0]

                route_times.append(total)

            # ==========================================
            # NEW: Print Savings Initialization Results
            # ==========================================
            print(f"Savings initialization produced {len(final_routes)} routes.")
            print("Initial routes from Savings:")
            for r in final_routes:
                print(r)

            # Formatting times to integers if they are whole numbers to match your C++ output
            formatted_times = [int(t) if float(t).is_integer() else round(t, 2) for t in route_times]
            print(f"route times: {formatted_times}")
            print("-" * 40)

            return final_routes, route_times

        return final_routes, route_times

    # ==========================================================
    # SELECT NEXT NODE (Eq 6–7 style probability rule)
    # ==========================================================
    def _select_next(self, curr, visited_comps):

        candidates = []
        weights = []

        for node in self.nodes:  # self.nodes are boundary nodes

            if node == curr:
                continue

            if self.comp_map[node] in visited_comps:
                continue

            # From base
            if curr == self.base:
                pher = self.pheromones.get((self.base, node), 0.0)
                heur = self.heuristics.get((self.base, node), 0.0)

            # Boundary to boundary
            else:
                if (curr, node) not in self.dist_matrix:
                    continue

                pher = self.pheromones.get((curr, node), 0.0)
                heur = self.heuristics.get((curr, node), 0.0)

            weight = (pher ** self.alpha) * (heur ** self.beta)

            if weight > 0:
                candidates.append(node)
                weights.append(weight)

        if not candidates:
            return None

        total = sum(weights)

        if total == 0:
            return random.choice(candidates)

        probabilities = [w / total for w in weights]

        return random.choices(candidates, weights=probabilities, k=1)[0]

    # ==========================================================
    # CONSTRUCT SOLUTION (UNCHANGED — already correct)
    # ==========================================================
    def construct_solution(self):

        pos = [self.base] * self.m
        times = [0.0] * self.m
        routes = [[self.base] for _ in range(self.m)]

        visited_comps = {self.artificial_base_comp_id}
        all_real_comps = set(self.comp_map.values())

        while len(visited_comps) < len(all_real_comps) + 1:

            moves = []

            for ant_idx in range(self.m):
                curr = pos[ant_idx]
                next_node = self._select_next(curr, visited_comps)

                if next_node is None:
                    continue

                if curr == self.base:
                    ts = self.base_dists[next_node]
                    tr = 0.0
                else:
                    ts, tr = self.dist_matrix[(curr, next_node)]

                repair_cost = tr if self.scenario == 2 else 0
                denom = times[ant_idx] + ts + repair_cost + 0.001
                char_val = 1.0 / denom

                moves.append((ant_idx, next_node, ts, tr, char_val))

            if not moves:
                break

            total_char = sum(m[4] for m in moves)
            probs = [m[4] / total_char for m in moves]
            choice = random.choices(moves, weights=probs, k=1)[0]

            ant, node, ts, tr, _ = choice

            pos[ant] = node
            routes[ant].append(node)

            cost = ts + (tr if self.scenario == 2 else 0)
            times[ant] += cost

            visited_comps.add(self.comp_map[node])

        for i in range(self.m):
            curr = pos[i]
            if curr != self.base:
                times[i] += self.base_dists[curr]
            routes[i].append(self.base)

        return routes, times

    # ==========================================================
    # RUN (Properly Seeded from Savings)
    # ==========================================================
    def run(self, iterations=100, num_of_colonies=30):

        best_routes = self.best_routes
        best_times = self.best_times
        best_times_sorted = sorted(self.best_times, reverse=True)

        Q = 100.0
        TAU_MIN = 0.01
        TAU_MAX = 1.0

        for _ in range(iterations):

            iter_solutions = [self.construct_solution()
                              for _ in range(num_of_colonies)]

            batch_best = min(iter_solutions,
                             key=lambda x: sorted(x[1], reverse=True))

            batch_routes, batch_times = batch_best

            if sorted(batch_times, reverse=True) < best_times_sorted:
                best_times_sorted = sorted(batch_times, reverse=True)
                best_routes = batch_routes
                best_times = batch_times

            # Evaporation
            for k in self.pheromones:
                self.pheromones[k] *= (1.0 - self.rho)

            # Choose global or iteration best
            if random.random() < 0.5:
                selected_routes = best_routes
                selected_times = best_times
            else:
                selected_routes = batch_routes
                selected_times = batch_times

            L_selected = sum(selected_times)
            delta_tau = Q / (L_selected + 1e-6)

            for r in selected_routes:
                for i in range(len(r) - 1):
                    u, v = r[i], r[i + 1]
                    if u == self.base or v == self.base:
                        continue

                    if (u, v) in self.pheromones:
                        self.pheromones[(u, v)] += delta_tau
                    if (v, u) in self.pheromones:
                        self.pheromones[(v, u)] += delta_tau

            # MMAS Clamping
            for k in self.pheromones:
                self.pheromones[k] = max(TAU_MIN,
                                         min(TAU_MAX, self.pheromones[k]))

        return best_routes, best_times


# ==========================================
# NEW: Visualize repaired edges in path
# ==========================================
def mark_repaired_edges(path, blocked_lookup):
    """
    Inserts {-} before broken edges for visualization
    Example:
    [15,10,9,{-},5,...]
    """

    marked_path = [path[0]]

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_key = tuple(sorted((u, v)))

        if edge_key in blocked_lookup:
            marked_path.append("{-}")  # mark repair

        marked_path.append(v)

    return marked_path
    

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # File Paths
    NODE_FILE = "SiouxFalls_node.tntp.txt"
    NET_FILE = "SiouxFalls_net.csv"
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
    # NEW: Print the Distance Matrix
    # ==========================================
    print("\n--- Reduced Graph Distance Matrix ---")
    for (u, v), (ts, tr) in sorted(dist_matrix.items()):
        print(f"Edge ({u:2}, {v:2}) -> Travel Time (TS): {ts:.2f}, Repair Time (TR): {tr:.2f}")
    print("-------------------------------------\n")

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

    

    # 4. Execute Scenarios
    NUM_UNITS = 4

    print("\n--- Running Scenario 1 (No Repair Cost) ---")
    aco1 = ModifiedACO(boundary_nodes, dist_matrix, comp_map, NUM_UNITS, base_node=base_node, G_full=G_full, scenario=1)
    routes1, times1 = aco1.run(iterations=100, num_of_colonies = 30)

    print("\nResults Scenario 1:")
    for i, (r, t) in enumerate(zip(routes1, times1)):
        # Calculate repair cost
        blocked_lookup = {
            tuple(sorted((x['u'], x['v']))): x['repair_time']
            for x in blocked_links_info
        }

        repaired_edges = set()  # resets per route

        full_path = get_full_path(G_full, r)

        ts_full, repair_cost = compute_path_TS_TR_with_repair(
            full_path, G_full, blocked_lookup, repaired_edges
        )

        marked_path = mark_repaired_edges(full_path, blocked_lookup)

        print(f"Unit {i + 1}: Time {ts_full:.2f}, Repair Cost {repair_cost:.2f}, Route: {r}")
        print(f"        Full Path: {marked_path}")

    # Convert the NumPy values to standard Python floats for a cleaner print
    clean_times = [float(t) for t in sorted(times1, reverse=True)]
    print(f"Sorted Times: {clean_times}")

    print("\n--- Running Scenario 2 (With Repair Cost) ---")
    aco2 = ModifiedACO(boundary_nodes, dist_matrix, comp_map, NUM_UNITS, base_node=base_node, G_full=G_full, scenario=2)
    routes2, times2 = aco2.run(iterations=100, num_of_colonies=30)

    print("\nResults Scenario 2:")
    for i, (r, t) in enumerate(zip(routes2, times2)):
        # Calculate repair cost
        blocked_lookup = {
            tuple(sorted((x['u'], x['v']))): x['repair_time']
            for x in blocked_links_info
        }

        repaired_edges = set()  # resets per route

        full_path = get_full_path(G_full, r)

        ts_full, repair_cost = compute_path_TS_TR_with_repair(
            full_path, G_full, blocked_lookup, repaired_edges
        )

        marked_path = mark_repaired_edges(full_path, blocked_lookup)

        print(f"Unit {i + 1}: Time {ts_full:.2f}, Repair Cost {repair_cost:.2f}, Route: {r}")
        print(f"        Full Path: {marked_path}")

    # Convert the NumPy values to standard Python floats for a cleaner print
    clean_times = [float(t) for t in sorted(times2, reverse=True)]
    print(f"Sorted Times: {clean_times}")
    for t in times2:
        print(t)

    