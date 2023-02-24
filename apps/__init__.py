import logging
import os
from uuid import uuid4
import numpy as np
from flask import Flask,session
from flask_session import Session
from flask_socketio import SocketIO

fname_Qfun = os.getcwd() + '\\apps\\static\\PursuitGame\\Qfunctions.npz'

# fname_Qfun = 'C:\\Users\\mason\\Desktop\\ResearchWebsite-master\\apps\\static\\PursuitGame\\Qfunctions.npz'
Qfunctions = np.load(fname_Qfun)

DEBUG = False

###############################################
#          Define flask app                   #
###############################################
app = Flask(__name__)
APP_KEY = '92659'
app.debug = DEBUG
app.config['SECRET_KEY'] = APP_KEY
app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_PERMANENT"] = True; logging.warning('Session is  permanant')
app.config[ "SESSION_TYPE"] = "filesystem"  # We set the session type to the filesystem,
app.config['SECRET_KEY'] = uuid4().hex  # Configure secret key for encryption
app.config["TEMPLATES_AUTO_RELOAD"] = True
Session(app)
socketio = SocketIO(app)

###############################################
#          Render Home page                   #
###############################################

def create_app():
    from . import config

    config.initialize(app,socketio)
    socketio.run(app, host="192.168.0.137", port=8080, debug=True, use_reloader=False)
    # app.run(debug=True)
    return app
