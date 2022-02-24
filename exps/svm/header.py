import os, sys

dirname = os.path.abspath(os.path.dirname(__file__))
os.chdir(dirname)
sys.path.append(os.path.join(dirname, "..", "shared_scripts"))

import include_implicit
