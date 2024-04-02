import uvicorn
from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import io
from PIL import Image
import os
import shutil
from fastapi.responses import RedirectResponse
from lungcolon import test_image
# from chat import collect_messages_text
app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name = "static")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
context = []
templates = Jinja2Templates(directory="templates")
data = {}
class Message(BaseModel):
    content: str
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message:Message):
    user_message = message.content
    print(context)
    response = collect_messages_text(user_message,context)
    return {"message": response}

@app.post("/upload")
async def upload_image(request: Request, name: str = Form(...),image: UploadFile = File(...)):
    if image is not None and image.filename != '':
        data["name"] = name 
        # Saving the uploaded image to the specified directory
        image_path = os.path.join('static', image.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        d = test_image(image_path)
        data.update(d)
        
        
        # Return a response with the processed image (and any other data)
        return d



if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=8000)


   #apt-get install -y libgl1-mesa-dev
   #sudo apt-get install libgl1-mesa-glx
   #pip install python-multipart
   #python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl