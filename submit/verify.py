import os
import time
import nbformat
from nbconvert import PythonExporter, HTMLExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from traitlets.config import Config
import warnings
warnings.filterwarnings('ignore') 

# Convert notebook to python
assignment_path = '../Assignment_1.ipynb'
export_path = 'solution.py'
template_path = 'Template.ipynb'
submission_path = '../Submission.html'

# Clean up
if os.path.exists(export_path):
    os.remove(export_path)

# Start Notebook -> Python conversion
print("Converting to python: {0}".format(assignment_path))
exporter = PythonExporter()

# source is a tuple of python source code, meta contains metadata
(source, meta) = exporter.from_filename(assignment_path)

# skip plotting calls in the python code to speed up loading
new_source_lines = []
new_source_lines.append("#!/usr/bin/env python")
for line in source.split('\n'):
    if line.startswith("plot_"):
        line = "# {}".format(line)
    if line.startswith("get_ipython"):
        line = "# {}".format(line)
    new_source_lines.append(line)
source = "\n".join(new_source_lines)

with open(export_path, 'w+') as fh:
    fh.writelines(source)
    fh.writelines("last_edit = '{}'".format(meta['metadata']['modified_date']))
while not os.path.exists(export_path):
    print("Waiting for []".format(export_path))
    time.sleep(1)    

# Run solution notebook
start = time.time()
with open(template_path) as f:
    snb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=2000, kernel_name='python3')

try:
    print("Running notebook... (may take a while)")
    out = ep.preprocess(snb, {'metadata': {'path': './'}})
except CellExecutionError:
    out = None
    msg = 'Error executing the notebook "%s".\n\n' % template_path
    msg += 'See notebook "%s" for the traceback.' % template_path
    print(msg)
    raise
finally:
    # Save notebook
    with open(template_path, mode='w', encoding='utf-8') as f:
        nbformat.write(snb, f)

# Export as HTML (PDF is too much hassle)
print("All good. Building report.")
c = Config()
c.TagRemovePreprocessor.enabled=True
c.TagRemovePreprocessor.remove_input_tags = set(["hide_input"])
c.preprocessors = ["TagRemovePreprocessor"]

html_exporter = HTMLExporter(config=c)
html_data, resources = html_exporter.from_notebook_node(snb)
html_data = html_data.replace('</head>', '<style>pre{font-family: "Times New Roman", Times, serif;}</style></head>')

with open(submission_path, "wb") as f:
    f.write(html_data.encode('utf8'))
    f.close()

print("Done in {} seconds".format(time.time() - start))
