import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from noteagent.core.folder_navigation import load_vault
from noteagent.core.chroma_db import ObsidianChromaDB
from noteagent.core.langchain_rag import traditional_obsidian_retrieval_chain

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

VAULT_DIR = Path.cwd() / "data/test_vault/kepano-obsidian-main"
POLL_INTERVAL_SEC = 60  # how often to check for changes if no watcher
DEFAULT_K = 7  # retrieval k
DEFAULT_LLM = "qwen3:14b"


class AgentLoop:
    def __init__(self):
        self.vault_path = VAULT_DIR
        self.vector_db: Optional[ObsidianChromaDB] = None
        self.rag_chains = []
        self.last_full_index = None
        self.running = False

    def initialize(self):
        print("[Agent] Loading vault...")
        all_notes = load_vault(self.vault_path)

        print("[Agent] Initializing vector store...")
        self.vector_db = ObsidianChromaDB(all_notes)

        if not self.vector_db.check_if_existing_vectorstorage():
            print("[Agent] No documents found — creating full index...")
            self.vector_db.create_new_note_index()
        else:
            print("[Agent] Existing index found — checking for updates...")

        print("[Agent] Building RAG chain...")
        self.rag_chains.append(traditional_obsidian_retrieval_chain(self.vector_db, llm_model=DEFAULT_LLM))

        self.last_full_index = time.time()
        print("[Agent] Ready.")

    def handle_user_query(self, query: str):
        if not query.strip():
            return

        print(f"\n[You] {query}")

        try:
            start = time.time()
            answer = self.rag_chains[0].invoke(query)
            elapsed = time.time() - start
            print(f"[Agent] ({elapsed:.1f}s)\n{answer}\n")
        except Exception as e:
            print(f"[Error] {e}")

    def handle_command(self, cmd: str) -> bool:
        cmd = cmd.strip().lower()
        if cmd in ("exit", "quit", "q"):
            print("[Agent] Shutting down...")
            return False

        elif cmd == "reindex":
            print("[Agent] Full re-index requested...")
            self.vector_db.create_new_note_index()
            self.last_full_index = time.time()
            print("[Agent] Re-index complete.")

        elif cmd == "status":
            status = self.vector_db.get_status()
            print("[Status]")
            for k, v in status.items():
                print(f"  {k}: {v}")

        elif cmd.startswith("set k="):
            try:
                new_k = int(cmd.split("=")[1])
                print(f"[Agent] Changing retrieval k to {new_k}")
                # Re-create chain with new k
                self.rag_chains = traditional_obsidian_retrieval_chain(self.vector_db, llm_model=DEFAULT_LLM)
            except:
                print("[Error] Invalid k value")

        else:
            print("Unknown command. Try: reindex, status, set k=10, exit")

        return True

    def run(self):
        self.initialize()
        self.running = True

        print("\nAgent loop started. Commands: reindex, status, set k=NN, exit")
        print("Or just type any question about your notes.\n")

        while self.running:
            try:
                user_input = input("> ").strip()

                if user_input.startswith(":"):
                    # command mode
                    keep_running = self.handle_command(user_input[1:])
                    if not keep_running:
                        break
                else:
                    # normal query
                    self.handle_user_query(user_input)

                # Optional: poll for vault changes every few cycles
                time.sleep(0.5)  # small delay to avoid CPU spin

            except KeyboardInterrupt:
                print("\n[Agent] Interrupted. Shutting down...")
                break
            except Exception as e:
                print(f"[Unexpected error] {e}")
                time.sleep(2)  # backoff

        print("[Agent] Loop ended.")


if __name__ == "__main__":
    agent = AgentLoop()
    agent.run()
