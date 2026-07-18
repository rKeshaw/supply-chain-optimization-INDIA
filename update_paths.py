import json
import searoute

# Load nodes to get coordinates
with open('data/nodes.json', 'r') as f:
    nodes_list = json.load(f)
nodes = {n['id']: n for n in nodes_list}

# Load edges
with open('data/edges.json', 'r') as f:
    edges = json.load(f)

# Update edges
updated = 0
for edge in edges:
    if edge.get('mode') == 'sea' and edge['from_id'] in nodes and edge['to_id'] in nodes:
        from_node = nodes[edge['from_id']]
        to_node = nodes[edge['to_id']]
        if 'lat' in from_node and 'lon' in from_node and 'lat' in to_node and 'lon' in to_node:
            try:
                route = searoute.searoute([from_node['lon'], from_node['lat']], [to_node['lon'], to_node['lat']])
                if route and "geometry" in route and "coordinates" in route["geometry"]:
                    edge['path'] = route["geometry"]["coordinates"]
                    updated += 1
            except Exception as e:
                print(f"Error calculating route for {edge['id']}: {e}")

# Save edges
with open('data/edges.json', 'w') as f:
    json.dump(edges, f, indent=4)

print(f"Successfully updated {updated} sea edges with pre-calculated paths.")
