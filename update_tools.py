import re

tools_data = {
    "code_search": ('code_search', 'CodeSearchInput', True, 'Search the codebase for relevant functions, classes, or logic using keywords or natural language.'),
    "file_read": ('file_read', 'FileReadInput', True, 'Read the content of a file. Supports line ranges for large files.'),
    "file_edit": ('file_edit', 'FileEditInput', False, 'Edit a file by replacing an exact string with a new string. This is safer than overwriting the whole file.'),
    "file_write": ('file_write', 'FileWriteInput', False, 'Create a new file or completely overwrite an existing file with new content.'),
    "multi_file_edit": ('multi_file_edit', 'MultiFileEditInput', False, 'Apply multiple surgical replacements to a single file in one go. Much more efficient than multiple file_edit calls.'),
    "grep_tool": ('grep_search', 'GrepInput', True, 'Search for a pattern across the codebase using regex. Returns file paths and matching lines.'),
    "bash_tool": ('bash', 'BashInput', False, 'Execute a shell command. Use this for running tests, build scripts, or git commands.'),
    "glob_tool": ('glob', 'GlobInput', True, 'Search for files using glob patterns. Returns a list of relative paths.'),
    "git_status": ('git_status', 'GitStatusInput', True, "Run 'git status' and return the result. Use to check which files are modified or untracked."),
    "git_diff": ('git_diff', 'GitDiffInput', True, "Run 'git diff' to see changes in tracked files. Supports diffing against a specific commit or staged changes."),
    "git_commit": ('git_commit', 'GitCommitInput', False, "Stage changes and create a git commit. Must provide a descriptive commit message."),
    "git_log": ('git_log', 'GitLogInput', True, "View the git commit history. Supports limiting the number of entries."),
    "web_search": ('web_search', 'WebSearchInput', True, "Perform a Google search to find information outside the codebase."),
    "web_fetch": ('web_fetch', 'WebFetchInput', True, "Fetch a URL and extract its main content as Markdown. Ideal for reading documentation."),
    "brief_tool": ('brief', 'BriefInput', True, "Generate a high-level summary of a file's structure (functions, classes, imports)."),
    "symbol_search": ('symbol_search', 'SymbolSearchInput', True, "Search for a specific code symbol (class or function) definition across the codebase."),
    "agent_delegate": ('agent_delegate', 'AgentDelegateInput', False, "Spawn a sub-agent to handle a complex sub-task. Returns the agent's final report."),
    "ask_user": ('ask_user', 'AskUserInput', True, "Ask the user a question to clarify requirements or get feedback."),
    "update_plan": ('update_plan', 'UpdatePlanInput', False, "Update the agent's current task plan and strategy."),
    "set_status": ('set_status', 'SetStatusInput', False, "Update the agent's current status message (what it is doing right now)."),
    "switch_model": ('switch_model', 'SwitchModelInput', True, "Switch the active LLM brain. Ideal for scaling intelligence up or down."),
    "undo_last_edit": ('undo_last_edit', 'UndoInput', False, "Roll back file changes made in a specific turn. Use the tool_use_id of the turn to revert."),
    "notebook_edit": ('notebook_edit', 'NotebookEditInput', False, "Surgically edit, insert, or delete Jupyter Notebook (.ipynb) cells."),
    "system_doctor": ('system_doctor', 'DoctorInput', True, "Audit system health (binaries, network, workspace toxicity). Run if tools are failing."),
    "read_url": ('read_url', 'WebFetchInput', True, "Fetch and distill web documentation or articles into Markdown."),
    "cost_report": ('cost_report', 'CostInput', True, "Show the current session's token usage and USD cost report.")
}

files_to_process = [
    'new_sysy/file_tools.py',
    'new_sysy/bash_tool.py',
    'new_sysy/misc_tools.py',
    'new_sysy/git_tools.py',
    'new_sysy/web_tools.py',
    'new_sysy/agent_tools.py'
]

for file_path in files_to_process:
    with open(file_path, 'r') as f:
        content = f.read()

    # Move current_engine imports to top
    if 'from tool_registry import current_engine' in content:
        content = re.sub(r'^[ \t]*from tool_registry import current_engine\n', '', content, flags=re.MULTILINE)

    # Add tool_registry imports
    if 'from tool_registry import' in content:
        # Avoid duplicate imports
        if 'register_tool' not in content:
            content = re.sub(r'from tool_registry import (.*)', r'from tool_registry import register_tool, \1', content, count=1)
        if 'current_engine' not in content and file_path in ['new_sysy/agent_tools.py', 'new_sysy/misc_tools.py']:
            content = re.sub(r'from tool_registry import (.*)', r'from tool_registry import current_engine, \1', content, count=1)
    else:
        imports = ['register_tool']
        if file_path in ['new_sysy/agent_tools.py', 'new_sysy/misc_tools.py']:
            imports.append('current_engine')
        content = f"from tool_registry import {', '.join(imports)}\n" + content

    for func_name, (name, schema, is_read_only, desc) in tools_data.items():
        pattern = r'^def ' + re.escape(func_name) + r'\b'
        if re.search(pattern, content, flags=re.MULTILINE):
            decorator = f'@register_tool(name="{name}", description="{desc}", input_schema={schema}, is_read_only={is_read_only})\n'
            content = re.sub(r'(^def ' + re.escape(func_name) + r'\b)', decorator + r'\1', content, flags=re.MULTILINE)
            
    with open(file_path, 'w') as f:
        f.write(content)

