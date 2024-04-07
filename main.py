from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
import io
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import google.generativeai as genai
import psycopg2


conn = psycopg2.connect(
    dbname="sample_db",
    user="app",
    password="247E5Zb8p5uQ1Ca89rPxld9k",
    host="informally-sought-honeybee.a1.pgedge.io",
    port="5432"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.get("/PatientForm")
async def Patient_form(request: Request):
    return templates.TemplateResponse("PatientForm.html", {"request": request})

@app.get("/report")
async def report_fun(request: Request):
    return templates.TemplateResponse("report.html", {"request": request})



@app.post("/upload")
async def report(request: Request, file: UploadFile = File(...),patientName: str = Form(...) ,dob: str = Form(...), gender: str=Form(...), email: str=Form(...)):
    s_img = await file.read()
    # Convert the bytes data to a NumPy array
    image = Image.open(io.BytesIO(s_img))

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    loaded_model = tf.keras.models.load_model('Lung.h5', compile=False)
    classes = {0: ('ca', 'colon adenocarcinoma'), 1: ('cb', 'colon benign'), 2: ('lac', 'lung adenocarcinoma'), 3: ('lb', 'lung benign'),
            4: ('lscc', 'lung squamous cell carcinoma'), 5: ('nc', 'Free from Cancer')}
    predictions = loaded_model.predict(img_array)
    max_prob = np.max(predictions)
    class_ind = np.argmax(predictions)
    class_name = classes[class_ind]

    img_base64 = base64.b64encode(s_img).decode('utf-8')
    result = {
        "img": img_base64,
        "prediction": class_name
        
    }
    return templates.TemplateResponse("PatientForm.html", {"request": request,  "img": img_base64, "result":class_name, "patientName": patientName,"dob":dob, "gender":gender, "email":email})


@app.get("/chat", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get_gemini_completion")
def get_gemini_completion(
                            gemini_api_key: str =Form(...),
                            prompt: str = Form(...),  
                        ):
    try:
        genai.configure(api_key = gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))