from fastapi import FastAPI

app = FastAPI()
processor = DocumentProcessor()

@app.post("/assess_quality")
async def assess_quality(content: str):
    return {"score": processor._assess_quality([content])}

@app.post("/process")
async def process_file(file: UploadFile):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    return processor.process_document(temp_path)
