class DeltaUpdater:
    def __init__(self):
        self.version_store = {}
        
    def get_changes(self, new_content: str, doc_id: str) -> Dict:
        old_content = self.version_store.get(doc_id, "")
        differ = difflib.Differ()
        diff = list(differ.compare(old_content.splitlines(), new_content.splitlines()))
        
        changes = {
            "added": [line[2:] for line in diff if line.startswith('+ ')],
            "removed": [line[2:] for line in diff if line.startswith('- ')]
        }
        
        self.version_store[doc_id] = new_content
        return changes
