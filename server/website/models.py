from . import db
from flask_login import UserMixIn

class User(db.Model, UserMixIn):
    id = db.Column(db.Integer, primary_key = True)
    email = db.Column(db.String(150), unique = True)
    password = db.Column(db.String(150))
    user_name = db.Column(db.String(150))
