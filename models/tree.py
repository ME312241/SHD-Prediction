import os

def save_tree(startpath, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = '  ' * level
            f.write(f'{indent}{os.path.basename(root)}/\n')
            subindent = '  ' * (level + 1)
            for file in files[:10]:  # Limit to first 10 files per folder for readability
                f.write(f'{subindent}{file}\n')
            if len(files) > 10:
                f.write(f'{subindent}... and {len(files) - 10} more files\n')

save_tree('.', 'tree_structure.txt')
print('Tree structure saved to tree_structure.txt')