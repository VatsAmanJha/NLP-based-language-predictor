import pickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load the trained pipeline
with open('language_classifier_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    prediction = pipeline.predict([text])[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "text": text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
