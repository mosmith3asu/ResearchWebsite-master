from flask import Flask,render_template,request
from flask_bootstrap import Bootstrap
from .setup import nav
from importlib import import_module



###############################################
#          Define flask app                   #
###############################################
app = Flask(__name__)



###############################################
#          Render Home page                   #
###############################################

def register_blueprints(app):
    for module_name in ('setup',):
        module = import_module('apps.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)



def create_app():
    Bootstrap(app)
    nav.init_app(app)

    register_blueprints(app)
    app.run(debug=True)
    return app
