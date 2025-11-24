import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from model import AIClassificationModel

app = FastAPI(title="AI Classification Service")

class AI_API:
    def __init__(self):
        self.classifier = AIClassificationModel()

    @app.post("/predict")
    async def predict(self, file: UploadFile = File(...)):
        if not file.content_type.startswith("image/"): # type: ignore
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")
        
        try:
            image_bytes = await file.read()
            result = self.classifier.predictFromBytes(image_bytes)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not process image")
        
        return result
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
