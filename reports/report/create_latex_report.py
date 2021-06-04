import os.path
import sys

path = os.path.dirname(__file__)
sys.path.insert(0,os.path.split(os.path.split(path)[0])[0])

from src.notebook_to_latex import convert_notebook_to_latex


notebook_path = os.path.join(path, '01.1.report.ipynb')
parent_path = os.path.split(path)[0]

build_directory = os.path.join(parent_path,'report_latex')

if not os.path.exists(build_directory):
    os.mkdir(build_directory)

skip_figures=False
convert_notebook_to_latex(notebook_path=notebook_path, build_directory=build_directory, save_main=True, skip_figures=skip_figures)