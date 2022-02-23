import os, sys

dirname = os.path.abspath(os.path.dirname(__file__))
os.chdir(dirname)
sys.path.append(os.path.join(dirname, "..", "..", "test"))

import include_implicit
