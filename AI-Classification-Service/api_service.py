import uvicorn
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from model import AIClassificationModel

app = FastAPI(title="AI Classification Service")
classifier = AIClassificationModel()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"): # type: ignore
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    try:
        image_bytes = await file.read()
        result = classifier.predictFromBytes(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image")
    
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}
    
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("POST", "8000"))
    uvicorn.run(app, host=host, port=port)
