from flask import Flask,request
from flask import render_template
from PIL import Image
import numpy as np
from model import model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/files")
def files():
    return render_template("pass.html")


@app.route("/upload", methods=['GET','POST'])
def upload():
    imagefile = request.files.get("imagefile")
    arr = np.array(imagefile)
    Image.open(imagefile).show()
    """
    modell = model("labels.txt","custom-model.txt")
    labels = modell.labels
    modell.execute(imagefile)
    label = modell.labeled()"""

    #print(type(labels),labels)
    

    if imagefile:
        return "label"
    else:
        return "false"



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
