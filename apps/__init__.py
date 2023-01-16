from flask import Flask,render_template,request
# from importlib import import_module

from flask_bootstrap import Bootstrap
from flask import Flask, render_template,session,request,redirect
from flask_session import Session
from flask_socketio import SocketIO
from uuid import uuid4

DEBUG = False

###############################################
#          Define flask app                   #
###############################################
app = Flask(__name__)

APP_KEY = '92659'
app.debug = DEBUG
app.config['SECRET_KEY'] = APP_KEY
app.config["SESSION_PERMANENT"] = False
app.config[ "SESSION_TYPE"] = "filesystem"  # We set the session type to the filesystem,
app.config['SECRET_KEY'] = uuid4().hex  # Configure secret key for encryption
app.config["TEMPLATES_AUTO_RELOAD"] = True



Session(app)
socketio = SocketIO(app)

###############################################
#          Render Home page                   #
###############################################

# def register_blueprints(app):
#     app.register_blueprint(blueprint)
#     # for module_name in ('setup',):
#     #     module = import_module('apps.{}.routes'.format(module_name))
#     #     app.register_blueprint(module.blueprint)
#





def create_app():
    from .setup import blueprint,nav
    app.register_blueprint(blueprint)
    # app.add_url_rule('/', 'show_user', show_user)
    # app.route(blueprint)
    Bootstrap(app)
    nav.init_app(app)

    # register_blueprints(app)
    #app.run(debug=True)
    # socketio.init_app(app)
    socketio.run(app, host="192.168.0.137", port=8080, debug=True, use_reloader=False)
    # app.run(debug=True)
    return app
