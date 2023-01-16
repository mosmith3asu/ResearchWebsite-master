from .navigation import init_nav
# from .routes import blueprint
# from . import routes, events
from flask import Blueprint

blueprint_name = 'pages_blueprint'
blueprint = Blueprint(blueprint_name,  __name__, url_prefix='')

# from .routes import blueprint

from . import routes,events
nav = init_nav()

