from flask import Flask , request , render_template, redirect
#from PIL import Image
from model import Model
from PIL import Image
from db import DataBase
import numpy as np
import cv2

app=Flask(__name__)

db=DataBase("db.json")
imageLocation=""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload",methods=['GET','POST'])
def upload():
    global db
    global imageLocation

    imagefile = request.files.get("imagefile")
    print(imagefile)

    modell = Model()
    #modell.detect(imagefile)

    im1=Image.open(imagefile)
    #im1.show()
    im1=np.array(im1)
    modell.detect(im1)
    #redirect("ocalhost:5000/image")

    if imagefile:
        return str(np.array(im1))#str(type(imagefile))#render_template("image.html",image_dir="deneme.jpg") #str(imagefile) + "\n\n*******\n\n" + str(labels)

    else:
        return "imagefile FALSE"

@app.route("/database")
def database():
    global db
    #arr=np.array(db.db.all())
    print(len(db.db.all()))
    #arr.reshape((1,len(db.db.all())))
    return str(db.db.all())



@app.route("/image")
def returnImage():
    return """
        <HTML>
        <BODY>
        <img src={}>
        </BODY>
        </HTML>
    """.format(imageLocation)


if __name__ == "__main__":
    app.run(debug=True)
