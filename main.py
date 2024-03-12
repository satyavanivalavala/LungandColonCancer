from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
import io
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.post("/report")
async def report(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    # Convert the bytes data to a NumPy array
    nparr = np.frombuffer(data, np.uint8)
    # Decode the image using cv2.imdecode
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (28, 28))
    
    model = tf.keras.models.load_model("./best_model.h5")
    result = model.predict(img_resized.reshape(1, 28, 28, 3))

    max_prob = max(result[0])
    classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}

    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind]
    print(class_name)
    _, img_encoded = cv2.imencode('.png', img_resized)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    result = {
        "img": img_base64,
        "prediction": class_name
    }
    return templates.TemplateResponse("base.html", {"request": request,  "img": img_base64, "result":class_name })