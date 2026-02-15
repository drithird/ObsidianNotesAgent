from pathlib import Path
from noteagent.core.obsidian_nav import ingest_plain_text
from noteagent.core.folder_navigation import load_vault


all_notes = load_vault(Path(Path.cwd() / "data/test_vault/kepano-obsidian-main"))
db = ingest_plain_text(all_notes)

query = "Tell me about Evergreen Notes"
results = db.similarity_search(query, k=5)

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:10000]}...")
