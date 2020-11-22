""" Application boostrap """
import os
from flask import Flask
from flask_smorest import Api
from bson import ObjectId

from face_detector.routes import bp

app = Flask("pulse-data")
app.config["OPENAPI_VERSION"] = "3.0.2"
app.config["OPENAPI_URL_PREFIX"] = "api"


# openapi
api = Api(app)
api.register_blueprint(bp)
# Map to ('string', 'ObjectId') passing type and format
api.register_field(ObjectId, "string", "ObjectId")
