def _chunk_content(self, text: str) -> List[str]:
    """混合分块策略"""
    # 结构化文档分块（Markdown/HTML）
    if self._is_structured(text):
        return self._structure_based_chunking(text)
    return self._semantic_chunking(text)

def _is_structured(self, text: str) -> bool:
    """检测结构化文档"""
    return any(tag in text for tag in ["# ", "## ", "<h1>", "<div"])

def _structure_based_chunking(self, text: str) -> List[str]:
    """基于文档结构分块"""
    chunks = []
    current_chunk = []
    
    for line in text.split("\n"):
        if re.match(r"^(#+ |<h\d>)", line.strip()):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    return chunks
