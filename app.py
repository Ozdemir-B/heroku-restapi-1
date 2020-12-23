from flask import Flask , request , render_template
#from PIL import Image
from model import model

app=Flask(__name__)

@app.route("/")
def home():
    return "home"

@app.route("/upload",methods=['GET','POST'])
def upload():
    imagefile = request.files.get("imagefile")

    modell = model()
    labels=modell.labels

    if imagefile:
        return str(imagefile) + "\n\n*******\n\n" + str(labels)

    else:
        return "imagefile FALSE"


if __name__ == "__main__":
    app.run()
