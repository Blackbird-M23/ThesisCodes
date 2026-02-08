import pandas as pd
import networkx as nx
import numpy as np
import random
from sklearn.cluster import KMeans

# ==========================================
# 1. DATA LOADING & PARSING (From main.py)
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
# 2. DISASTER SIMULATION (From main.py)
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
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    node_file = "SiouxFalls_node.tntp.txt"
    net_file = "SiouxFalls_net.tntp.txt"
    output_file = "blocked_links_generated1.csv"

    print("Loading Sioux Falls Data...")
    try:
        nodes_df, links_df = load_sioux_falls_data(net_file, node_file)
    except FileNotFoundError:
        print("Error: Could not find Sioux Falls data files.")
        exit(1)

    print("Generating Disaster Scenario using Clustering...")
    # You can change n_components to generate different scenarios
    G, nodes_info, blocked_links, comp_map = create_disaster_scenario(nodes_df, links_df, n_components=5)

    print(f"Generated {len(blocked_links)} blocked links across {len(set(comp_map.values()))} components.")

    # Convert blocked links to DataFrame for saving
    # Format expected by main2.py: NodeA, NodeB, RepairTime
    data = []
    for link in blocked_links:
        data.append({
            'NodeA': link['u'],
            'NodeB': link['v'],
            'RepairTime': link['repair_time']
        })
    
    df_out = pd.DataFrame(data)
    df_out.to_csv(output_file, index=False)
    
    print(f"Dataset saved to '{output_file}'.")
    print("You can use this file as input for main2.py (rename to blocked_links.csv or update main2.py).")
