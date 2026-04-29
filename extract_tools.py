import ast

with open('old_registry.py', 'r') as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', '') == '_UNSORTED_TOOLS':
        dict_node = node.value
        for key, value in zip(dict_node.keys, dict_node.values):
            if isinstance(value, ast.Call) and getattr(value.func, 'id', '') == 'build_tool':
                kwargs = {kw.arg: kw.value for kw in value.keywords}
                name = ast.literal_eval(kwargs['name'])
                func = getattr(kwargs['func'], 'id', None)
                schema = getattr(kwargs['input_schema'], 'id', None)
                desc = ast.literal_eval(kwargs['description'])
                read_only = ast.literal_eval(kwargs.get('is_read_only', ast.Constant(value=False)))
                print(f"{func}|{name}|{schema}|{read_only}|{desc}")
