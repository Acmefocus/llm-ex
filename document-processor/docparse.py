def _process_csv(self, path: str) -> Tuple[str, Dict]:
    """处理CSV文件"""
    df = pd.read_csv(path)
    return df.to_markdown(), {"rows": len(df), "columns": list(df.columns)}

def _process_docx(self, path: str) -> Tuple[str, Dict]:
    """处理Word文档"""
    from docx import Document
    doc = Document(path)
    content = [p.text for p in doc.paragraphs]
    return "\n".join(content), {"paragraphs": len(content)}
