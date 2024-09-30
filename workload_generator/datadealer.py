import json
import numpy as np

with open('/root/Faper/workload/STATS/train.json', 'r') as file:
    query_plan = json.load(file)

operations = ['Hash Join', 'Merge Join', 'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Index Scan', 'Bitmap Heap Scan']
# 43 列
dbinfo = {
    "ph": ["posthistorytypeid", "userid", "postid", "id", "creationdate"],
    "p": ["id", "score", "answercount", "favoritecount", "creationdate", "commentcount", "posttypeid", "owneruserid", "lasteditoruserid", "viewcount"],
    "u": ["id", "views", "downvotes", "upvotes", "creationdate", "reputation"],
    "c": ["id", "creationdate", "score", "userid", "postid"],
    "pl": ["postid", "id", "relatedpostid", "linktypeid", "creationdate"],
    "t": ["excerptpostid", "id", "count"],
    "b": ["userid", "date", "id"],
    "v": ["userid", "postid", "votetypeid", "creationdate", "bountyamount", "id"]
}

table_info = {
    "u": [
        ["id", -1, 55747],
        ["reputation", 1, 87393],
        ["views", 0, 20932],
        ["upvotes", 0, 11442],
        ["downvotes", 0, 1920],
        ["creationdate", "2010-07-19 06:55:26", "2014-09-14 01:01:44"]
    ],
    "p": [
        ["id", 1, 115378],
        ["posttypeid", 1, 7],
        ["score", -19, 192],
        ["viewcount", 1, 175495],
        ["owneruserid", -1, 55746],
        ["answercount", 0, 136],
        ["commentcount", 0, 45],
        ["favoritecount", 0, 233],
        ["lasteditoruserid", -1, 55733],
        ["creationdate", "2009-02-02 14:21:12", "2014-09-14 02:09:23"]
    ],
    "pl": [
        ["id", 108, 3356789],
        ["postid", 4, 115360],
        ["relatedpostid", 1, 115163],
        ["linktypeid", 1, 3],
        ["creationdate", "2010-07-21 14:47:33", "2014-09-13 20:54:31"]
    ],
    "ph": [
        ["id", 1, 386848],
        ["posthistorytypeid", 1, 38],
        ["postid", 1, 115378],
        ["userid", -1, 55746],
        ["creationdate", "2009-02-02 14:21:12", "2014-09-14 02:54:13"]
    ],
    "c": [
        ["id", 1, 221292],
        ["postid", 1, 115376],
        ["score", 0, 90],
        ["userid", 3, 55746],
        ["creationdate", "2009-02-02 14:45:19", "2014-09-14 02:04:27"]
    ],
    "v": [
        ["id", 1, 386258],
        ["postid", 1, 115376],
        ["votetypeid", 1, 16],
        ["userid", -1, 55706],
        ["bountyamount", 0, 500],
        ["creationdate", "2009-02-02 00:00:00", "2014-09-14 00:00:00"]
    ],
    "b": [
        ["id", 1, 92240],
        ["userid", 2, 55746],
        ["date", "2010-07-19 19:39:07", "2014-09-14 02:31:28"]
    ],
    "t": [
        ["id", 1, 1869],
        ["count", 1, 7244],
        ["excerptpostid", 2331, 114058],
        []
    ]
}

column_names = ["ph.posthistorytypeid", "ph.userid", "ph.postid", "ph.id", "ph.creationdate", "p.id", "p.score", "p.answercount", "p.favoritecount", "p.creationdate", "p.commentcount", "p.posttypeid", "p.owneruserid", "p.lasteditoruserid", "p.viewcount", "u.id", "u.views", "u.downvotes", "u.upvotes", "u.creationdate", "u.reputation", "c.id", "c.creationdate", "c.score", "c.userid", "c.postid", "pl.postid", "pl.id", "pl.relatedpostid", "pl.linktypeid", "pl.creationdate", "t.excerptpostid", "t.id", "t.count", "b.userid", "b.date", "b.id", "v.userid", "v.postid", "v.votetypeid", "v.creationdate", "v.bountyamount", "v.id"]
# One-hot 编码操作
def encode_operation(node_type):
    encoded = [0] * len(operations)
    index = operations.index(node_type)
    encoded[index] = 1
    return encoded, index

# One-hot 编码 join 涉及的列
def encode_join_columns(hash_cond):
    hash_cond = hash_cond.lstrip("(").rstrip(")")
    encoded = [0] * 43
    if hash_cond and "=" in hash_cond:
        parts = hash_cond.split("=")
        if parts[0].strip() in column_names:
            index = column_names.index(parts[0].strip())
            encoded[index] = 1
        if parts[1].strip() in column_names:
            index = column_names.index(parts[1].strip())
            encoded[index] = 1
    return encoded

# One-hot 编码 filter 涉及的列
def encode_filter_columns(alias, filter):
    encoded = [0] * 43
    filter = filter.lstrip("(").rstrip(")")
    parts = filter.split(" ")
    column_name = f"{alias}.{parts[0].strip()}"
    if column_name in column_names:
        index = column_names.index(column_name)
        encoded[index] = 1
    return encoded

# One-hot 编码 filter 涉及的过滤符号
def encode_filter_symbols(filter_expr):
    symbols = ['<', '>', '=']
    encoded = [0] * len(symbols)
    if filter_expr:
        for symbol in symbols:
            if symbol in filter_expr:
                index = symbols.index(symbol)
                encoded[index] = 1
                return encoded

# 归一化参数表示过滤条件
def normalize_filter(filter_expr, table):
    filter_expr = filter_expr.lstrip("(").rstrip(")")
    parts = filter_expr.split()
    value = float(parts[-1].lstrip("\'").rstrip("\'::integer"))
    columns = table_info.get(table)
    for i in range(len(columns)):
        if columns[i][0] == parts[0]:
            min = columns[i][1]
            max = columns[i][2]
            value = (value-min)/(max-min)
            return [value]
    assert False
    

def encode_plan_node(node):
    node_type = node["Node Type"]
    if node_type not in operations:
        if "Plans" in node:
            assert len(node['Plans']) == 1

            return encode_plan_node(node['Plans'][0])
            
        else:
            assert False
    
    tree_node = {}
    join_flag = False

    operation_encoding, num = encode_operation(node_type)
    if num == 0: # Hash Join
        join_flag = True

        join_columns_encoding = []
        join_columns_encoding = encode_join_columns(node["Hash Cond"])
        filter_columns_encoding = [0] * 43
        filter_symbols_encoding = [0] * 3
        filter_normalized_value = [0.00]
    else:
        if num == 1:  # Merge Join
            join_flag = True

            join_columns_encoding = []
            join_columns_encoding = encode_join_columns(node["Merge Cond"])
            filter_columns_encoding = [0] * 43
            filter_symbols_encoding = [0] * 3
            filter_normalized_value = [0.00]
        else:  # scan

            join_columns_encoding = [0] * 43
            filter_columns_encoding = [0] * 43
            filter_symbols_encoding = [0] * 3
            filter_normalized_value = [0.00]

            if 'Filter' in node:
                filter_columns_encoding = encode_filter_columns(node["Alias"], node["Filter"])
                filter_symbols_encoding = encode_filter_symbols(node["Filter"])
                filter_normalized_value = normalize_filter(node["Filter"], node["Alias"])
            if 'Index Cond' in node:
                filter_columns_encoding = encode_filter_columns(node["Alias"], node["Index Cond"])
                filter_symbols_encoding = encode_filter_symbols(node["Index Cond"])
                filter_normalized_value = normalize_filter(node["Index Cond"], node["Alias"])
            if 'Recheck Cond' in node:
                filter_columns_encoding = encode_filter_columns(node["Alias"], node["Recheck Cond"])
                filter_symbols_encoding = encode_filter_symbols(node["Recheck Cond"])
                filter_normalized_value = normalize_filter(node["Recheck Cond"], node["Alias"])

    encoded_node = {
        "operation_encoding": operation_encoding,
        "join_columns_encoding": join_columns_encoding,
        "filter_columns_encoding": filter_columns_encoding,
        "filter_symbols_encoding": filter_symbols_encoding,
        "filter_normalized_value": filter_normalized_value
    }

    tree_node['embedding'] = encoded_node
    tree_node['card'] = node["Actual Rows"]*node["Actual Loops"]

    if join_flag:
        assert "Plans" in node
        assert len(node["Plans"]) == 2
        tree_node['children'] = [encode_plan_node(sub_node) for sub_node in node["Plans"]]

    return tree_node

def preorder_traversal_with_id(tree, id_counter=0):
    tree['node_id'] = id_counter
    id_counter += 1
    if 'children' in tree:
        for child in tree['children']:
            id_counter = preorder_traversal_with_id(child, id_counter)
    return id_counter

all_trees_result = []
for plan in query_plan:
    tree = encode_plan_node(plan[0]["Plan"])
    preorder_traversal_with_id(tree)
    all_trees_result.append(tree)

with open('output.json', 'w') as f:
    json.dump(all_trees_result, f)