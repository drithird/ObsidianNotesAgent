from pathlib import Path
from noteagent.core.folder_navigation import walk


def test_walk():

    print(walk(Path.cwd().parent, 0, 8))


test_walk()
