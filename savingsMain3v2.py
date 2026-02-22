import pandas as pd
import networkx as nx
import random


# .\.venv\Scripts\Activate.ps1  : activate

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
    # blocked_edges = []

    # Helper to check if edge exists in CSV
    # We use a set for faster lookup of (u, v) pairs
    user_blocks = set()
    block_info = {}  # Store repair times, used DICTIONARY

    for _, row in blocked_df.iterrows():
        u, v = int(row['NodeA']), int(row['NodeB'])
        repair_time = float(row['RepairTime'])

        key = tuple(sorted((u, v)))
        user_blocks.add(key)  # user_defined_blocks
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
        # 1. Boundary <-> Boundary: h_ij = 1 / (TS_ij + epsilon)
        for i in self.nodes:
            for j in self.nodes:
                if i != j and (i, j) in self.dist_matrix:
                    ts, _ = self.dist_matrix[(i, j)]
                    self.heuristics[(i, j)] = 1.0 / (ts + 0.001)

        # 2. Base → Boundary: h_0j = 1 / (TS_0j + epsilon)
        for bn in self.nodes:
            ts = self.base_dists[bn]
            self.heuristics[(self.base, bn)] = 1.0 / (ts + 0.001)

        # -------------------------------
        # Savings Initialization
        # -------------------------------
        # We use the robust version of the savings algorithm to get an initial solution.
        # This provides us with a good starting point for the ACO and allows us to initialize pheromones based on a reasonable solution quality.
        
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
        # It returns 2 things:
        # final_routes: list of routes (one route per repair unit after merging)
        # route_times: total travel time for each route (used for ACO initialization)

        # For every boundary node, create a tiny route: base → node → base
        routes = {node: [self.base, node, self.base] for node in self.nodes}
        #Tracks which route each node currently belongs to. Initially, each node is in its own route.
        node_route_map = {node: node for node in self.nodes}

        savings = []

        for i in self.nodes:
            for j in self.nodes:
                # Skip same node and non-existent edges in reduced graph
                if i == j: 
                    continue
                if (i, j) not in self.dist_matrix:
                    continue
                # distance from base to i and j
                d0i = self.base_dists[i]
                d0j = self.base_dists[j]
                # distance between i and j in reduced graph (travel time only)
                dij = self.dist_matrix[(i, j)][0]

                Sij = d0i + d0j - dij
                savings.append((Sij, i, j)) 

        savings.sort(reverse=True, key=lambda x: x[0]) # we will consider Highest savings value first (best merges first).

        for Sij, i, j in savings:

            ri_id = node_route_map[i]
            rj_id = node_route_map[j]

            if ri_id == rj_id:
                continue

            ri = routes[ri_id]
            rj = routes[rj_id]
            # condition : 
            # i must be at the end of its route (before base) and 
            # j must be at the start of its route (after base) to merge them without creating loops
            if ri[-2] == i and rj[1] == j: # example: route_i = [0, 5, 0] and route_j = [0, 8, 0] and we want to merge on i=5 and j=8 → merged = [0, 5, 8, 0]
                merged = ri[:-1] + rj[1:] # removing the last base from route_i and the first base from route_j, then concatenating them.

                routes[ri_id] = merged
                del routes[rj_id]

                for node in merged:
                    if node != self.base: # the base node is not part of the mapping since it's not a boundary node, we only map the boundary nodes to their route IDs
                        node_route_map[node] = ri_id 

            if len(routes) <= self.m:
                break

        final_routes = list(routes.values())
        print(f"Savings initialization produced {len(final_routes)} routes.")
        print("Initial routes from Savings:")
        for r in final_routes:
            print(r)

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
        print("route times:", route_times)
        
        return final_routes, route_times

    # ==========================================================
    # SELECT NEXT NODE (Eq 6–7 style probability rule)
    # ==========================================================
    def _select_next(self, curr, visited_comps):

        candidates = []
        weights = []

        for node in self.nodes:

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
# 4. ACO ALGORITHM
# ==========================================

# class ModifiedACO:
#     def __init__(self, boundary_nodes, dist_matrix, node_comp_map, num_units, base_node, G_full, scenario=1):
#         self.nodes = boundary_nodes
#         self.dist_matrix = dist_matrix
#         self.comp_map = node_comp_map
#         self.m = num_units
#         self.scenario = scenario
#         self.G_full = G_full
#
#         # ACO Parameters
#         self.alpha = 1.0
#         self.beta = 1.0
#         self.rho = 0.2
#
#         # State
#         self.heuristics = {}
#         self.pheromones = {}
#
#         # -------------------------------
#         # USER-DEFINED BASE STATION
#         # -------------------------------
#         if base_node not in self.comp_map:
#             raise ValueError(f"Base node {base_node} does not exist in the network.")
#
#         self.base = base_node
#         self.artificial_base_comp_id = -1
#
#         # Pre-calculate Base -> Boundary Distances (Travel Time only)
#         # Using intact graph (G_full)
#         lengths, _ = nx.single_source_dijkstra(self.G_full, self.base, weight='weight')
#         self.base_dists = {}
#         for bn in self.nodes:
#             self.base_dists[bn] = lengths.get(bn, float('inf'))
#
#         print(f"Base station set to node {self.base} (Artificial Component {self.artificial_base_comp_id})")
#
#         # Init Pheromones/Heuristics
#         # 1. Boundary <-> Boundary
#         # First create pheromone keys
#         for i in self.nodes:
#             for j in self.nodes:
#                 if i != j:
#                     self.pheromones[(i, j)] = 0.0
#
#         # --------- INITIAL SOLUTION USING SAVINGS ---------
#         init_routes, init_times = self.find_initial_solution()
#         self.best_routes = init_routes
#         self.best_times = init_times
#
#         lengthpath = sum(init_times)
#
#         TAU_MIN = 0.01
#         TAU_MAX = 1.0
#
#         tau0 = 1.0 / (lengthpath + 1e-6)
#         tau0 = max(TAU_MIN, min(TAU_MAX, tau0))
#
#         for key in self.pheromones:
#             self.pheromones[key] = tau0
#
#         # 2. Base -> Boundary (One-way for selection)
#         for bn in self.nodes:
#             ts = self.base_dists[bn]
#             h = 1.0 / (ts + 0.001) if ts < float('inf') else 0.0
#             self.heuristics[(self.base, bn)] = h
#             self.pheromones[(self.base, bn)] = 0.5
#
#
#     def _select_next(self, curr, visited_comps):
#         # Build probabilities
#         candidates = []
#         weights = []
#
#         # CASE 1: Start at Base
#         # if curr == self.base:
#         #     targets = self.nodes
#         # else:
#         #     # CASE 2: Boundary to Boundary
#         #     targets = self.nodes
#         targets = self.nodes
#
#         for node in targets:
#             if node == curr: continue
#             if node == self.base:
#                 continue
#
#             target_comp = self.comp_map[node]
#             if target_comp not in visited_comps:
#
#                 # Fetch Pheromone & Heuristic
#                 if curr == self.base:
#                     ph = self.pheromones.get((self.base, node), 0.1)
#                     he = self.heuristics.get((self.base, node), 0.0001)
#                 else:
#                     # Validate connectivity in reduced graph
#                     if (curr, node) not in self.dist_matrix:
#                         continue
#                     ph = self.pheromones.get((curr, node), 0.1)
#                     he = self.heuristics.get((curr, node), 0.0001)
#
#                 weight = (ph ** self.alpha) * (he ** self.beta)
#
#                 candidates.append(node)
#                 weights.append(weight)
#
#         if not candidates: return None
#
#         # Roulette
#         total = sum(weights)
#         if total == 0: return random.choice(candidates)
#         probs = [w / total for w in weights]
#         return random.choices(candidates, weights=probs, k=1)[
#             0]  # through choosing between the probs we are choosing a candidates
#
#     #--------- Savings Algo------------#
#     def find_initial_solution(self):
#         """
#         Clarke & Wright Savings algorithm adapted to reduced graph.
#         Returns: routes (list of routes), times (list of route times)
#         """
#
#         # Step 1: Start with trivial routes: base -> node -> base
#         routes = []
#         for node in self.nodes:
#             routes.append([self.base, node, self.base])
#
#         # Step 2: Compute savings S_ij = c(0,i) + c(0,j) - c(i,j)
#         savings = []
#
#         for i in self.nodes:
#             for j in self.nodes:
#                 if i == j:
#                     continue
#
#                 if (i, j) in self.dist_matrix:
#                     c0i = self.base_dists[i]
#                     c0j = self.base_dists[j]
#                     cij = self.dist_matrix[(i, j)][0]
#
#                     s = c0i + c0j - cij
#                     savings.append((s, i, j))
#
#         # Step 3: Sort savings descending
#         savings.sort(reverse=True)
#
#         # Step 4: Merge routes greedily
#         for s, i, j in savings:
#
#             route_i = None
#             route_j = None
#
#             # Find routes containing i and j
#             for r in routes:
#                 if r[-2] == i:
#                     route_i = r
#                 if r[1] == j:
#                     route_j = r
#
#             # Merge condition
#             if route_i and route_j and route_i != route_j:
#                 # Merge route_i and route_j
#                 new_route = route_i[:-1] + route_j[1:]
#
#                 routes.remove(route_i)
#                 routes.remove(route_j)
#                 routes.append(new_route)
#
#             # Stop when number of routes equals number of repair units
#             if len(routes) <= self.m:
#                 break
#
#         # Step 5: If still more routes than repair units,
#         # merge smallest ones arbitrarily
#         while len(routes) > self.m:
#             r1 = routes.pop()
#             routes[0] = routes[0][:-1] + r1[1:]
#
#         # Step 6: Compute times
#         times = []
#         for r in routes:
#             total_time = 0.0
#
#             for k in range(len(r) - 1):
#                 u, v = r[k], r[k + 1]
#
#                 if u == self.base:
#                     total_time += self.base_dists[v]
#                 elif v == self.base:
#                     total_time += self.base_dists[u]
#                 else:
#                     ts, tr = self.dist_matrix[(u, v)]
#                     total_time += ts
#
#             times.append(total_time)
#
#         return routes, times
#     # def savings_init(self): replacing the name
#     # def  find_initial_solution(self):
#     #     """
#     #     Clarke & Wright savings initialization
#     #     tailored to reduced graph.
#     #     Returns:
#     #         initial_routes (list of routes),
#     #         route_times (list of travel times only)
#     #     """
#     #
#     #     # Step 1: Start with one route per boundary node
#     #     routes = {node: [self.base, node, self.base] for node in self.nodes}
#     #
#     #     # Track which route each node belongs to
#     #     node_route_map = {node: node for node in self.nodes}
#     #
#     #     # Step 2: Compute savings for all valid boundary pairs
#     #     savings = []
#     #
#     #     for i in self.nodes:
#     #         for j in self.nodes:
#     #             if i == j:
#     #                 continue
#     #             if (i, j) not in self.dist_matrix:
#     #                 continue
#     #
#     #             d_base_i = self.base_dists[i]
#     #             d_base_j = self.base_dists[j]
#     #             d_ij = self.dist_matrix[(i, j)][0]
#     #
#     #             Sij = d_base_i + d_base_j - d_ij
#     #             savings.append((Sij, i, j))
#     #
#     #     # Step 3: Sort savings descending
#     #     savings.sort(reverse=True, key=lambda x: x[0])
#     #
#     #     # Step 4: Merge routes greedily
#     #     for Sij, i, j in savings:
#     #
#     #         route_i_id = node_route_map[i]
#     #         route_j_id = node_route_map[j]
#     #
#     #         if route_i_id == route_j_id:
#     #             continue  # already same route
#     #
#     #         route_i = routes[route_i_id]
#     #         route_j = routes[route_j_id]
#     #
#     #         # Check merge condition:
#     #         # i must be at end of its route (before base)
#     #         # j must be at start of its route (after base)
#     #         if route_i[-2] == i and route_j[1] == j:
#     #
#     #             # Merge route_i + route_j
#     #             new_route = route_i[:-1] + route_j[1:]
#     #
#     #             # Update route storage
#     #             routes[route_i_id] = new_route
#     #             del routes[route_j_id]
#     #
#     #             # Update node mapping
#     #             for node in new_route:
#     #                 if node != self.base:
#     #                     node_route_map[node] = route_i_id
#     #
#     #     # Step 5: Extract final routes
#     #     final_routes = list(routes.values())
#     #     if not final_routes:
#     #         raise RuntimeError("Savings initialization produced no routes.")
#     #
#     #     # Step 6: If more routes than repair units, merge smallest ones
#     #     while len(final_routes) > self.m:
#     #         # merge last two smallest routes
#     #         r1 = final_routes.pop()
#     #         r2 = final_routes.pop()
#     #         merged = r1[:-1] + r2[1:]
#     #         final_routes.append(merged)
#     #
#     #     # Step 7: Compute route travel times
#     #     route_times = []
#     #     for route in final_routes:
#     #         total_time = 0.0
#     #         for k in range(len(route) - 1):
#     #             u, v = route[k], route[k + 1]
#     #
#     #             if u == self.base:
#     #                 total_time += self.base_dists[v]
#     #             elif v == self.base:
#     #                 total_time += self.base_dists[u]
#     #             else:
#     #                 total_time += self.dist_matrix[(u, v)][0]
#     #
#     #         route_times.append(total_time)
#     #
#     #     return final_routes, route_times
#     #---------------#
#
#     def construct_solution(self):
#         # Each ant starts at base
#         pos = [self.base] * self.m  # All ants start from base , just a normal LIST
#         times = [0.0] * self.m  # Time spent by each ant
#         routes = [[self.base] for _ in range(self.m)]  # A LIST of LISTS
#
#         visited_comps = {self.artificial_base_comp_id}
#         # We need to visit all REAL components
#         all_real_comps = set(self.comp_map.values())
#
#         # Step-by-step construction
#         # Stop when we have visited all real components + the artificial base
#         while len(visited_comps) < len(all_real_comps) + 1:
#
#             # Identify valid moves for ALL ants
#             moves = []  # This will store all valid moves ants can make at this step.
#
#             for ant_idx in range(self.m):
#                 curr_node = pos[ant_idx]
#                 target = self._select_next(curr_node, visited_comps)
#
#                 if target is not None:
#                     # Determine cost
#                     if curr_node == self.base:
#                         # Base -> Boundary (Travel Time only, No Repair)
#                         ts = self.base_dists[target]
#                         tr = 0.0
#                     else:
#                         # Boundary -> Boundary
#                         # Explicit lookup, no fallbacks
#                         if (curr_node, target) in self.dist_matrix:
#                             ts, tr = self.dist_matrix[(curr_node, target)]
#                         else:
#                             # Should not happen given _select_next checks, but careful
#                             continue
#
#                     # Cost calculation based on Scenario
#                     repair_cost = tr if self.scenario == 2 else 0
#
#                     # Eq 13: Attractiveness char
#                     # Note: We add a small epsilon to avoid div by zero if times are 0
#                     denom = times[ant_idx] + ts + repair_cost + 0.001
#                     char_val = 1.0 / denom
#
#                     moves.append({
#                         'ant': ant_idx,
#                         'target': target,
#                         'ts': ts,
#                         'tr': tr,
#                         'char': char_val
#                     })
#
#             if not moves: break  # If no ant cant find the next node to go to -> moves list will be empty after all the iteration → stop
#
#             # Probabilistic selection of WHICH ANT moves
#             total_char = sum(m['char'] for m in moves)
#             probs = [m['char'] / total_char for m in moves]
#
#             choice = random.choices(moves, weights=probs, k=1)[
#                 0]  # through choosing between the probs we are choosing a move
#             # only one ant move each iteration NOT all the ants
#
#             # Apply Move
#             ant = choice['ant']
#             target = choice['target']
#
#             pos[ant] = target  # current position of ant after the move
#             routes[ant].append(target)  # adding the new position of that chosen ant
#
#             # Update Time
#             cost = choice['ts'] + (choice['tr'] if self.scenario == 2 else 0)
#             times[ant] += cost
#
#             # Update Visited
#             visited_comps.add(self.comp_map[target])
#
#         # Return to base logic (FORCED)
#         for i in range(self.m):
#             curr = pos[i]
#
#             if curr == self.base:
#                 ts = 0.0
#                 tr = 0.0
#             else:
#                 # Return cost: Base Dist from current node + 0 repair
#                 ts = self.base_dists[curr]
#                 tr = 0.0
#
#             routes[i].append(self.base)
#             cost = ts + (tr if self.scenario == 2 else 0)
#             times[i] += cost
#
#         return routes, times
#
#     def run(self, iterations=100, num_of_colonies = 30):
#         # best_routes = None
#         # # Init best times with infinity
#         # best_times = [float('inf')] * self.m
#         # best_times_sorted = [float('inf')] * self.m
#         best_routes = self.best_routes
#         best_times = self.best_times
#         best_times_sorted = sorted(self.best_times, reverse=True)
#
#         # MMAS Constants
#         Q = 100.0
#         TAU_MIN = 0.01
#         TAU_MAX = 1.0
#
#         for _ in range(iterations):
#             # Run colony (10 ants/solutions per iter)
#             iter_solutions = [self.construct_solution() for _ in range(num_of_colonies)]
#
#             # Find best in this batch (Lexicographical Min)
#             # Sort times desc: [max_time, 2nd_max, ...]
#             # Python compares lists element by element, which matches lexicographical rank
#             batch_best_sol = min(iter_solutions, key=lambda x: sorted(x[1], reverse=True))
#             batch_routes, batch_times = batch_best_sol
#
#             # Compare with global best
#             if sorted(batch_times, reverse=True) < best_times_sorted:
#                 best_times_sorted = sorted(batch_times, reverse=True)
#                 best_routes = batch_routes
#                 best_times = batch_times
#
#             # ==========================================
#             # MMAS Pheromone Update Logic
#             # ==========================================
#
#             # 1. Evaporation (Apply to ALL edges first)
#             for k in self.pheromones:
#                 self.pheromones[k] *= (1.0 - self.rho)
#
#             # 2. Selection Strategy (Global vs Iteration Best)
#             # 50% probability for each. If global best not found yet (should be set), use batch.
#             selected_routes = None
#             selected_times = None
#
#             if best_routes is None:
#                 selected_routes = batch_routes
#                 selected_times = batch_times
#             else:
#                 if random.random() < 0.5:
#                     selected_routes = best_routes
#                     selected_times = best_times
#                 else:
#                     selected_routes = batch_routes
#                     selected_times = batch_times
#
#             # 3. Deposit Step
#             # L_selected = total completion time of selected solution
#             L_selected = sum(selected_times)
#
#             # Avoid division by zero
#             delta_tau = Q / (L_selected + 1e-6)
#
#             for r in selected_routes:
#                 for i in range(len(r) - 1):
#                     u, v = r[i], r[i + 1]
#
#                     # IMPORTANT: Do NOT deposit on edges involving base
#                     if u == self.base or v == self.base:
#                         continue
#
#                     # Update pheromone on edge (u, v) and symmetric (v, u)
#                     # Check existence to avoid key errors (though keys should exist for boundary nodes)
#                     key = (u, v)
#                     if key in self.pheromones:
#                         self.pheromones[key] += delta_tau
#
#                     key_rev = (v, u)
#                     if key_rev in self.pheromones:
#                         self.pheromones[key_rev] += delta_tau
#
#             # 4. MMAS Clamping
#             for k in self.pheromones:
#                 if self.pheromones[k] < TAU_MIN:
#                     self.pheromones[k] = TAU_MIN
#                 elif self.pheromones[k] > TAU_MAX:
#                     self.pheromones[k] = TAU_MAX
#                     # 5. MMAS Clamping
#                     # Ensure pheromones are within bounds
#                     # This is a safety check to prevent pheromones from going out of bounds
#                     # It should not be necessary if rho is properly set
#                     # self.pheromones[k] = max(TAU_MIN, min(TAU_MAX, self.pheromones[k]))
#
#         return best_routes, best_times


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
    routes1, times1 = aco1.run(iterations=100, num_of_colonies = 30)

    print("\nResults Scenario 1:")
    for i, (r, t) in enumerate(zip(routes1, times1)):
        # Calculate repair cost
        repair_cost = 0
        for idx in range(len(r) - 1):
            u, v = r[idx], r[idx + 1]
            if (u, v) in dist_matrix:
                _, tr = dist_matrix[(u, v)]
            else:
                tr = 0
            repair_cost += tr
        print(f"Unit {i + 1}: Time {t:.2f}, Repair Cost {repair_cost:.2f}, Route: {r}")
    print(f"Sorted Times: {sorted(times1, reverse=True)}")

    print("\n--- Running Scenario 2 (With Repair Cost) ---")
    aco2 = ModifiedACO(boundary_nodes, dist_matrix, comp_map, NUM_UNITS, base_node=base_node, G_full=G_full, scenario=2)
    routes2, times2 = aco2.run(iterations=100, num_of_colonies = 30)

    print("\nResults Scenario 2:")
    for i, (r, t) in enumerate(zip(routes2, times2)):
        # Calculate repair cost
        repair_cost = 0
        for idx in range(len(r) - 1):
            u, v = r[idx], r[idx + 1]
            if (u, v) in dist_matrix:
                _, tr = dist_matrix[(u, v)]
            else:
                tr = 0
            repair_cost += tr
        print(f"Unit {i + 1}: Time {t:.2f}, Repair Cost {repair_cost:.2f}, Route: {r}")
    print(f"Sorted Times: {sorted(times2, reverse=True)}")


