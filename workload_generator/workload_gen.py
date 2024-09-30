import json


def traverse_tree(tree):
    result = {
        'edges': [],
        'levels': [],
        'node': [],
        'feature': [],
        'label': []
    }

    def dfs(node, parent_id, parent_level, visit_order = 0):
        # node order
        if 'children' in node:
            visit_order+=1
            result['node'].append(visit_order)
        else:
            result['node'].append(0)
        # feature
        embedding_lists = [node['embedding']['operation_encoding'],
                          node['embedding']['join_columns_encoding'],
                          node['embedding']['filter_columns_encoding'],
                          node['embedding']['filter_symbols_encoding'],
                          node['embedding']['filter_normalized_value']]
        flat_embedding = []
        for sublist in embedding_lists:
            for item in sublist:
               flat_embedding.append(str(item))
        result['feature'].append(" ".join(flat_embedding))
        # label(card)
        result['label'].append(node.get('card'))

        current_id = node.get('node_id', None)
        current_level = parent_level + 1

        if parent_id is not None:
            result['edges'].append(f"{parent_id} {current_id}")

        result['levels'].append(current_level)

        children = node.get('children', [])
        for child in children:
            dfs(child, current_id, current_level, visit_order)

    dfs(tree, None, 0)

    return result

with open('/root/Faper/workload/STATS/output.json', 'r') as file:
    trees = json.load(file)

num = 1
for tree in trees:
    tree_info = traverse_tree(tree)
    edge_info = tree_info['edges']
    level_info = list(zip(tree_info['levels'][: - 1], tree_info['levels'][1:]))
    feature_info = tree_info['feature']
    node_info = tree_info['node']
    label_info = tree_info['label']

    with open('train/feature.txt', 'a') as f:
        f.write(f"plan {num}\n")
        for feature in feature_info:
            f.write(f"{feature}\n")

    with open('train/label.txt', 'a') as f:
        f.write(f"plan {num}\n")
        for label in label_info:
            f.write(f"{label}\n")

    with open('train/adjacency_list.txt', 'a') as f:
        f.write(f"plan {num}\n")
        for edge in edge_info:
            f.write(f"{edge}\n")

    with open('train/edge_order.txt', 'a') as f:
        f.write(f"plan {num}\n")
        if level_info:
            max_level = max([level for _, level in level_info])
            for _, level in level_info:
                f.write(f"{max_level + 1 - level} ")
        f.write('\n')

    with open('train/node_order.txt', 'a') as f:
        f.write(f"plan {num}\n")
        if node_info:
            max_level = max(node_info)
            for node in node_info:
                if node == 0:
                    f.write("0 ")
                else:
                    f.write(f"{max_level + 1 - node} ")
        f.write('\n')
    num += 1



