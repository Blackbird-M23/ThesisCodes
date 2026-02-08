# WITH ERROR HANDLING FOR BLOCKED LINKS

# import pandas as pd
# import networkx as nx
# import random
# import sys
#
#
# # ==========================================
# # 1. DATA LOADING & VALIDATION
# # ==========================================
#
# def load_sioux_falls_data(net_file, node_file):
#     # Load Nodes
#     nodes_df = pd.read_csv(node_file, sep='\t')
#     nodes_df.columns = [c.strip() for c in nodes_df.columns]
#
#     # Load Links - Skip metadata until header found
#     with open(net_file, 'r') as f:
#         lines = f.readlines()
#     header_line = next(i for i, line in enumerate(lines) if "init_node" in line or "From" in line)
#     links_df = pd.read_csv(net_file, skiprows=header_line, sep='\t')
#     links_df.columns = [c.strip() for c in links_df.columns]
#     return nodes_df, links_df
#
#
# def load_disaster_input(csv_file):
#     try:
#         df = pd.read_csv(csv_file)
#         if not {'NodeA', 'NodeB', 'RepairTime'}.issubset(df.columns):
#             raise ValueError("CSV must have NodeA, NodeB, and RepairTime columns.")
#         return df
#     except FileNotFoundError:
#         print(f"FATAL: {csv_file} not found in the directory.")
#         sys.exit(1)
#
#
# # ==========================================
# # 2. DISASTER MODELING
# # ==========================================
#
# def apply_disaster_scenario(nodes_df, links_df, blocked_df):
#     G_full = nx.Graph()
#     link_travel_times = {}
#
#     # Build Intact Graph
#     for _, row in links_df.iterrows():
#         u, v = int(row['init_node']), int(row['term_node'])
#         cost = row.get('Free_Flow_Time', row.get('Length', 1))
#         edge = tuple(sorted((u, v)))
#         link_travel_times[edge] = cost
#         G_full.add_edge(u, v, weight=cost)
#
#     # Process Blocks
#     user_blocks = {tuple(sorted((int(r.NodeA), int(r.NodeB)))): float(r.RepairTime) for _, r in blocked_df.iterrows()}
#
#     G_broken = G_full.copy()
#     actual_blocks = []
#     for u, v in list(G_full.edges()):
#         key = tuple(sorted((u, v)))
#         if key in user_blocks:
#             if G_broken.has_edge(u, v):
#                 G_broken.remove_edge(u, v)
#                 actual_blocks.append({'u': u, 'v': v, 'ts': link_travel_times[key], 'tr': user_blocks[key]})
#
#     # Component Check (Crucial for the paper's logic)
#     components = list(nx.connected_components(G_broken))
#     if len(components) < 2:
#         print("ERROR: Your blocked_links.csv did not disintegrate the network.")
#         print("The algorithm requires at least 2 isolated parts to find boundary nodes.")
#         sys.exit(1)
#
#     node_comp_map = {node: cid for cid, nodes in enumerate(components) for node in nodes}
#     print(f"SUCCESS: Network broken into {len(components)} isolated components.")
#     return G_full, actual_blocks, node_comp_map
#
#
# # ==========================================
# # 3. GRAPH REDUCTION & ACO
# # ==========================================
#
# def build_reduced_graph(G_full, blocked_info, node_comp_map):
#     boundary_nodes = set()
#     for b in blocked_info:
#         if node_comp_map[b['u']] != node_comp_map[b['v']]:
#             boundary_nodes.add(b['u'])
#             boundary_nodes.add(b['v'])
#
#     b_nodes = list(boundary_nodes)
#     if not b_nodes:
#         print("ERROR: No boundary nodes found. Ensure blocked links connect different components.")
#         sys.exit(1)
#
#     dist_matrix = {}
#     blocked_lookup = {tuple(sorted((x['u'], x['v']))): x['tr'] for x in blocked_info}
#
#     for start in b_nodes:
#         lengths, paths = nx.single_source_dijkstra(G_full, start, weight='weight')
#         for end in b_nodes:
#             if start == end: continue
#             path = paths[end]
#             ts_sum = lengths[end]
#             tr_sum = sum(blocked_lookup.get(tuple(sorted((path[i], path[i + 1]))), 0) for i in range(len(path) - 1))
#             dist_matrix[(start, end)] = (ts_sum, tr_sum)
#
#     return b_nodes, dist_matrix
#
#
# class ModifiedACO:
#     def __init__(self, boundary_nodes, dist_matrix, node_comp_map, m, scenario):
#         self.nodes = boundary_nodes
#         self.dist_matrix = dist_matrix
#         self.comp_map = node_comp_map
#         self.m = m
#         self.scenario = scenario
#         self.pheromones = {(i, j): 0.5 for i in self.nodes for j in self.nodes if i != j}
#         # Start at Node 1 if it's a boundary, else pick first available
#         self.base = 1 if 1 in self.nodes else self.nodes[0]
#
#     def construct_solution(self):
#         pos = [self.base] * self.m
#         times = [0.0] * self.m
#         routes = [[self.base] for _ in range(self.m)]
#         visited_comps = {self.comp_map[self.base]}
#         all_comps = set(self.comp_map.values())
#
#         while len(visited_comps) < len(all_comps):
#             moves = []
#             for ant_idx in range(self.m):
#                 curr = pos[ant_idx]
#                 for node in self.nodes:
#                     if self.comp_map[node] not in visited_comps:
#                         ts, tr = self.dist_matrix.get((curr, node), (999, 0))
#                         cost = ts + (tr if self.scenario == 2 else 0)
#                         char_val = 1.0 / (times[ant_idx] + cost + 0.1)
#                         moves.append({'ant': ant_idx, 'target': node, 'cost': cost, 'char': char_val})
#
#             if not moves: break
#             choice = random.choices(moves, weights=[m['char'] for m in moves], k=1)[0]
#             ant, target = choice['ant'], choice['target']
#             pos[ant] = target
#             routes[ant].append(target)
#             times[ant] += choice['cost']
#             visited_comps.add(self.comp_map[target])
#
#         # Return to base
#         for i in range(self.m):
#             ts, tr = self.dist_matrix.get((pos[i], self.base), (0, 0))
#             routes[i].append(self.base)
#             times[i] += (ts + (tr if self.scenario == 2 else 0))
#         return routes, times
#
#     def run(self, iters=50):
#         best_t_sorted = [float('inf')] * self.m
#         best_r = None
#         for _ in range(iters):
#             routes, times = self.construct_solution()
#             if sorted(times, reverse=True) < best_t_sorted:
#                 best_t_sorted = sorted(times, reverse=True)
#                 best_r = routes
#         return best_r, best_t_sorted
#
#
# # ==========================================
# # EXECUTION
# # ==========================================
# if __name__ == "__main__":
#     nodes_df, links_df = load_sioux_falls_data("SiouxFalls_net.tntp.txt", "SiouxFalls_node.tntp.txt")
#     blocked_df = load_disaster_input("blocked_links.csv")
#     G_full, blocks, comp_map = apply_disaster_scenario(nodes_df, links_df, blocked_df)
#     b_nodes, d_matrix = build_reduced_graph(G_full, blocks, comp_map)
#
#     for s in [1, 2]:
#         print(f"\n--- Scenario {s} ---")
#         aco = ModifiedACO(b_nodes, d_matrix, comp_map, 4, s)
#         routes, times = aco.run()
#         print(f"Best Lexicographical Times: {times}")