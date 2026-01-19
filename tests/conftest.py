# Solution to import helper script by this answer https://stackoverflow.com/a/33515264
import os
import sys

# Add the rutile-ai directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add the 'tests' directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tests")))
