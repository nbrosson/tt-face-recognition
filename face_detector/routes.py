from flask_smorest import Blueprint
from flask import request, jsonify, render_template
from .face_identifier import predict_input_identity
import json
import timeit

bp = Blueprint(__name__, "home")


@bp.route("/healthcheck")
def healthcheck():
    return "App running", 200


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/face_identification", methods=["POST"])  # it's actually not a POST as the operation is stateless
def face_identification():
    start = timeit.default_timer()
    image_data = request.files["input_image"].read()
    res = predict_input_identity(input=image_data)
    stop = timeit.default_timer()
    res["computing_time"] = stop-start
    response = json.dumps(res, sort_keys=True, indent=4, separators=(',', ': '))
    return render_template("result.html", res=res), 200
