from pathlib import Path
from typing import List, Dict, Any
import obsidiantools.api as otools
import pandas as pd


def walk(current: Path, max_depth: int, depth: int = 0) -> List[Dict]:
    """Recursively search for Obsidian vaults under the given path.

    Looks for directories containing a '.obsidian' subfolder and counts
    the number of .md files in each discovered vault.

    Args:
        current: The starting directory path to scan.
        depth: Current recursion depth (starts at 0).
        max_depth: Maximum depth to recurse into subdirectories.

    Returns:
        List of dictionaries, each describing a found vault with path,
        name, markdown file count, and discovery depth.

    Raises:
        PermissionError/OSError: Silently skipped if access denied.
    """
    vaults = []
    if depth > max_depth:
        return []
    try:
        if (current / ".obsidian").is_dir():
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


def unpack_note(
    vault: otools.Vault,
    abs_filepath: str,
    vault_root: Path,
    note_metadata_df: pd.DataFrame,
) -> Dict[str, Any]:
    note_path = Path(abs_filepath)
    if not note_path.is_file():
        return {"relative_path": "unknown", "error": "not_a_file"}

    rel_path = note_path.relative_to(vault_root).as_posix()

    # Fast stats
    row_match = note_metadata_df[note_metadata_df["abs_filepath"] == abs_filepath]
    base_stats = {}
    if not row_match.empty:
        row = row_match.iloc[0]
        base_stats = {
            "rel_filepath": row["rel_filepath"],
            "note_exists": bool(row["note_exists"]),
            "n_backlinks": int(row["n_backlinks"]),
            "n_wikilinks": int(row["n_wikilinks"])
            if pd.notna(row["n_wikilinks"])
            else 0,
            "n_tags": int(row["n_tags"]) if pd.notna(row["n_tags"]) else 0,
            "n_embedded_files": int(row["n_embedded_files"])
            if pd.notna(row["n_embedded_files"])
            else 0,
            "modified_time_iso": row["modified_time"].isoformat()
            if pd.notna(row["modified_time"])
            else None,
        }

    # Rich getters
    getters_data: Dict[str, Any] = {}
    try:
        getters_data.update(
            {
                "name": note_path.stem,
                "raw_content": vault.get_source_text(note_path.stem),
                "clean_body": vault.get_readable_text(note_path.stem),
                "frontmatter": vault.get_front_matter(note_path.stem),
                "tags": vault.get_tags(note_path.stem),
                "backlinks": list(vault.get_backlinks(note_path.stem)),
            }
        )
        if hasattr(vault, "get_embedded_files"):
            getters_data["embedded_files"] = list(
                vault.get_embedded_files(note_path.stem)
            )
        if hasattr(vault, "get_md_links"):
            getters_data["outgoing_md_links"] = list(vault.get_md_links(note_path.stem))
    except Exception as e:
        getters_data["getters_error"] = str(e)

    # Derived
    derived = {
        "is_isolated": base_stats.get("n_backlinks", 0) == 0
        and base_stats.get("n_wikilinks", 0) == 0,
        "approx_word_count": len(getters_data.get("clean_body", "").split()),
        "has_frontmatter": bool(getters_data.get("frontmatter", {})),
        "last_modified": base_stats.get("modified_time_iso"),
    }

    return {
        "relative_path": rel_path,
        **base_stats,
        **getters_data,
        **derived,
    }


def load_vault(vault_path: Path) -> List[Dict]:
    if not vault_path.is_dir():
        raise ValueError(f"Not a directory: {vault_path}")
    if not (vault_path / ".obsidian").is_dir():
        raise ValueError(f"No .obsidian folder found in {vault_path}")

    vault = otools.Vault(vault_path).connect().gather()
    note_metadata_df = vault.get_note_metadata()
    file_paths: List = note_metadata_df["abs_filepath"].copy().to_list()
    note_metadata_df["abs_filepath"] = note_metadata_df["abs_filepath"].astype(str)
    note_metadata_df["rel_filepath"] = note_metadata_df["rel_filepath"].astype(str)

    all_notes_data = []
    for note_path_str in vault.md_file_index.values():
        abs_path = str(vault_path / note_path_str)
        try:
            data = unpack_note(vault, abs_path, vault_path, note_metadata_df)
            all_notes_data.append(data)
        except Exception as e:
            print(f"Failed to unpack {note_path_str}: {e}")
            continue

    print(f"Processed {len(all_notes_data)} notes successfully")
    return all_notes_data


# Example usage
if __name__ == "__main__":
    vault_dir = Path(
        "/home/drithird/GenericProjects/ObsidianNotesAgent/data/test_vault/kepano-obsidian-main"
    )
    try:
        notes = load_vault(vault_dir)
        # Now you have list of dicts → ready for LangChain
        print(f"First note keys: {list(notes[0].keys()) if notes else 'No notes'}")
        yum = 1 + 1
    except Exception as e:
        print(f"Vault loading failed: {e}")
