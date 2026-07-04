import os, re
path = r'E:\school\python_school\boston_housing\visualizer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all plt.savefig calls to add os.makedirs before them
# Pattern: plt.savefig(path, ...)  ->  os.makedirs(...); plt.savefig(path, ...)
import re
old = 'plt.savefig(path, dpi=150, bbox_inches=\\'tight\\')'
new = 'os.makedirs(os.path.dirname(path), exist_ok=True); plt.savefig(path, dpi=150, bbox_inches=\\'tight\\')'
content = content.replace(old, new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Fixed visualizer.py - {content.count(new)} savefig calls updated')
