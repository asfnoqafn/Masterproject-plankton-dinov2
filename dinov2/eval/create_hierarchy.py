
import json
import pandas as pd

class HierarchyNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def is_descendant(self, node_name):
        """Check if a given node name is a descendant."""
        for child in self.children:
            if child.name == node_name or child.is_descendant(node_name):
                return True
        return False
    
def deserialize_hierarchy(data):
    """Reconstruct the hierarchy tree from a dictionary."""
    node = HierarchyNode(data["name"])
    for child_data in data["children"]:
        child_node = deserialize_hierarchy(child_data)
        node.add_child(child_node)
    return node

def load_hierarchy_from_file(file_path):
    """Load the hierarchy tree from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return deserialize_hierarchy(data)

def serialize_hierarchy(node):
    """Convert the hierarchy tree to a dictionary for JSON serialization."""
    return {
        "name": node.name,
        "children": [serialize_hierarchy(child) for child in node.children]
    }

def save_hierarchy_to_file(hierarchy_root, file_path):
    """Save the hierarchy tree to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(serialize_hierarchy(hierarchy_root), file, indent=4)

def find_node(node, target):
        if node.name == target:
            return node
        for child in node.children:
            result = find_node(child, target)
            if result:
                return result
        return None

def build_hierarchy(taxonomy_csv_path="dinov2/eval/taxa_zoo_scan.csv", lineage_column="lineage_level1", taxon_column="taxon_level1"):
    # Read taxonomy data
    df = pd.read_csv(taxonomy_csv_path)
    taxonomy_data = df.dropna(subset=[lineage_column, taxon_column])


    # Initialize root node and node map
    root = HierarchyNode("Root")
    node_map = {"Root": root}

    for _, row in taxonomy_data.iterrows():
        lineage = row[lineage_column].strip("/").split("/")  # Remove slashes and split
        taxon_name = row[taxon_column]  # Get the taxon name
        parent_name = "Root"

        for i, level in enumerate(lineage):
            # Use taxon name for the last node, otherwise use the lineage level
            node_name = taxon_name if i == len(lineage) - 1 else level
            node_name = ''.join(e for e in node_name if e.isalnum()).lower()  # Remove special characters and convert to lowercase
            # Check if the node already exists
            if node_name not in node_map:
                # Create the node and add it to its parent
                new_node = HierarchyNode(node_name)
                node_map[parent_name].add_child(new_node)
                node_map[node_name] = new_node

            # Update parent_name to the current node for the next iteration
            parent_name = node_name

    return root
# Build hierarchy
hierarchy_root = build_hierarchy()

# Save to file
save_hierarchy_to_file(hierarchy_root, "hierarchy_zoo_scan.json")