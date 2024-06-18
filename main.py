import base64
from io import BytesIO

import inference
from flask import Flask, render_template, request
from matplotlib.figure import Figure
from PIL import Image

app = Flask(__name__)
model = inference.get_roboflow_model("building-footprint-extract/3")


def process_image(image: Image) -> BytesIO:
    predictions = model.infer(image=image)[0].predictions
    fig = Figure()
    ax = fig.subplots()
    ax.axis("off")
    ax.imshow(image)
    for pred in predictions:
        points = pred.points
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        ax.scatter(xs, ys)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf


@app.route("/")
def upload():
    return render_template("file_upload_form.html")


@app.route("/success", methods=["POST"])
def success():
    if request.method != "POST":
        return
    f = request.files["file"]
    image = Image.open(f.stream)
    buf = process_image(image)
    img_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render_template("success.html", name=f.filename, img_data=img_data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
