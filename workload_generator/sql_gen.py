import random
import re

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

def has_element_equal(arr, table):
    for item in arr:
        if item == table:
            return True
    return False

def element_index(arr, table):
    index = -1
    for item in arr:
        index+=1
        if item == table:
            return index
    return index

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def add_random_filter():
    operators = ['<', '>', '=', '>=', '<=']
    return random.choice(operators)

def add_random_filter_1():
    operators = ['<', '>', '>=', '<=']
    return random.choice(operators)

def add_random_filter_2():
    operators = ['<', '>']
    return random.choice(operators)

def add_random_filter_3():
    operators = ['<', '>', '=']
    return random.choice(operators)

def add_random_disturbance(floating_point_number, max_disturbance=5):
    disturbance = random.uniform(-max_disturbance, max_disturbance)
    return floating_point_number + disturbance

def generate_random_number(m, n):
    return random.randint(m, n)


def generate_single_predicate_sqls(original_sql):
    parts = original_sql.split('WHERE')
    new_sql_nochange = parts[0] + "WHERE"
    where_clauses = parts[1].strip().rstrip(";").split(' AND ')

    join_clauses = []
    filter_clause = []
    for clause in where_clauses:
        names = (clause.split('=')[1]).strip()
        if names[0].isalpha():
            join_clauses.append(clause)
        else:
            filter_clause.append(clause)
    for clause in join_clauses:
        new_sql_nochange += (" AND " + clause)

    new_sqls = []
    # for clause in where_clauses:
    used_table = []
    # 基于原始数据给予扰动随机生成查询负载
    column = []
    filter = []
    bound = []

    for clause in filter_clause:
        if '!=' in clause:
            condition = clause.split('!=')
            if not is_number(condition[1].strip()):
                continue
            table = condition[0].split('.')[0]
            if has_element_equal(used_table, table):
                continue
            used_table.append(table)
            column.append(condition[0].strip())
            filter.append('!=')
            bound.append(float(condition[1].strip()))
            continue

        if '<=' in clause:
            condition = clause.split('<=')
            if not is_number(condition[1].strip()):
                continue
            table = condition[0].split('.')[0]
            if has_element_equal(used_table, table):
                continue
            used_table.append(table)
            column.append(condition[0].strip())
            filter.append('<=')
            bound.append(float(condition[1].strip()))
            continue

        if '>=' in clause:
            condition = clause.split('>=')
            if not is_number(condition[1].strip()):
                continue
            table = condition[0].split('.')[0]
            if has_element_equal(used_table, table):
                continue
            used_table.append(table)
            column.append(condition[0].strip())
            filter.append('>=')
            bound.append(float(condition[1].strip()))
            continue

        if '=' in clause:
            condition = clause.split('=')
            if not is_number(condition[1].strip()):
                continue
            table = condition[0].split('.')[0]
            if has_element_equal(used_table, table):
                continue
            used_table.append(table)
            column.append(condition[0].strip())
            filter.append('=')
            bound.append(float(condition[1].strip()))
            continue

        if '<' in clause:
            condition = clause.split('<')
            if not is_number(condition[1].strip()):
                continue
            table = condition[0].split('.')[0]
            if has_element_equal(used_table, table):
                continue
            used_table.append(table)
            column.append(condition[0].strip())
            filter.append('<')
            bound.append(float(condition[1].strip()))
            continue

        if '>' in clause:
            condition = clause.split('>')
            if not is_number(condition[1].strip()):
                continue
            table = condition[0].split('.')[0]
            if has_element_equal(used_table, table):
                continue
            used_table.append(table)
            column.append(condition[0].strip())
            filter.append('>')
            bound.append(float(condition[1].strip()))
            continue
            
    for j in range(40):
        new_sqls.append(new_sql_nochange)
        for i in range(len(column)):
            if filter[i] == '=':
                new_filter = add_random_filter()
                if new_filter == '=':
                    new_sqls[j] += (" AND " + column[i]+'='+str(bound[i]))
                else:
                    new_sqls[j] += (" AND " + column[i] + new_filter + str(add_random_disturbance(bound[i])))
            else:
                new_sqls[j] += (" AND " + column[i]+add_random_filter_1()+str(add_random_disturbance(bound[i])))
        new_sqls[j] += ";"
        new_sqls[j].replace("WHERE AND", "WHERE")
    return new_sqls

def generate_sqls(original_sql):
    parts = original_sql.split('WHERE')
    new_sql_nochange = parts[0] + "WHERE"
    where_clauses = parts[1].strip().rstrip(";").split(' AND ')

    join_clauses = []
    filter_clause = []
    join_table = []
    for clause in where_clauses:
        names = (clause.split('=')[1]).strip()
        if names[0].isalpha():
            join_clauses.append(clause)
        else:
            filter_clause.append(clause)
    for clause in join_clauses:
        new_sql_nochange += (" AND " + clause)
        table_name = clause.split('=')[0].split('.')[0].strip()
        if not has_element_equal(join_table, table_name):
            join_table.append(table_name)
        table_name = clause.split('=')[1].split('.')[0].strip()
        if not has_element_equal(join_table, table_name):
            join_table.append(table_name)

    new_sqls = []
    # for clause in where_clauses:
    table_equal = []
    clause_equal = []
    # 基于原始查询语句的join结构随机生成过滤谓词

    for clause in filter_clause:
        if '<=' in clause:
            continue
        if '>=' in clause:
            continue
        if '=' in clause:
            condition = clause.split('=')
            if not is_number(condition[1].strip()):
                continue
            table = condition[0].split('.')[0]
            table_equal.append(table.strip())
            clause_equal.append(clause.strip())
            
    for j in range(40):
        new_sqls.append(new_sql_nochange)
        for i in range(len(join_table)):
            table_name = join_table[i]
            if has_element_equal(table_equal, table_name):
                filter = add_random_filter_3()
                if filter == '=':
                    new_sqls[j] += (" AND " + clause_equal[element_index(table_equal, table_name)])
                else:
                    length = len(table_info.get(table_name))-1
                    column = table_info.get(table_name)[generate_random_number(0, length-1)]
                    new_sqls[j] += (" AND " + table_name + '.' + column[0] + filter + str(generate_random_number(column[1], column[2])))
                    # column[0] 表 column[1] min column[2] max
            else:
                filter = add_random_filter_2()
                length = len(table_info.get(table_name))-1
                column = table_info.get(table_name)[generate_random_number(0, length-1)]
                new_sqls[j] += (" AND " + table_name + '.' + column[0] + filter + str(generate_random_number(column[1], column[2])))
        new_sqls[j] += ";"
        new_sqls[j].replace("WHERE AND", "WHERE")
    return new_sqls


def process_sql_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                original_sql = line.strip()
                if original_sql:
                    new_sql_statements = generate_sqls(original_sql)
                    for new_sql in new_sql_statements:
                        output_file.write(new_sql + '\n')
    except FileNotFoundError:
        print(f"输入文件 {input_file_path} 不存在。")

input_sql_file_path = '/root/Faper/workload/STATS/stats_CEB.sql'
output_sql_file_path = '/root/Faper/workload/STATS/train.sql'
process_sql_file(input_sql_file_path, output_sql_file_path)