from pathlib import Path
import os
from typing import List, Dict
import logging


def walk(current: Path, depth: int, max_depth: int) -> List[Dict]:
    vaults = []
    print(f"Looking at the {current} directory")
    if depth > max_depth:
        return []
    try:
        if (current / ".obsidian").is_dir():
            print(f"Found .obsidian in {current}")
            md_count = sum(1 for p in current.rglob("*.md") if p.is_file())
            vaults.append(
                {
                    "path": current.resolve(),
                    "name": current.name or "[root]",
                    "md_count": md_count,
                    "depth": depth,
                }
            )

        for item in current.iterdir():
            if item.is_dir():
                vaults_to_append = walk(item, depth + 1, max_depth)
                if len(vaults_to_append) > 0:
                    vaults.extend(vaults_to_append)
    except (PermissionError, OSError):
        pass
    return [vault for vault in vaults]
