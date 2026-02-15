from pathlib import Path
from noteagent.core.folder_navigation import walk, load_vault


def test_walk():
    start_dir = Path.cwd().parent
    vaults = walk(start_dir, 8, 0)
    assert len(vaults) == 1, f"Expected 1 vault, found {len(vaults)}"

    vault = vaults[0]

    expected = (start_dir / "data/test_vault/kepano-obsidian-main").resolve().absolute()

    assert vault["path"] == expected
    assert vault["name"] == "kepano-obsidian-main"
    assert vault["md_count"] >= 103
    assert vault["depth"] == 3


def test_load_vault_obsidiantools():
    vault = walk(Path.cwd().parent, 8)
    obsidian_vault = load_vault(vault["path"])
    print(obsidian_vault)


test_walk()
