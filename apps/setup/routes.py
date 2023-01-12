from flask import Blueprint
from flask import render_template, request
# from flask_login import login_required
from jinja2 import TemplateNotFound

blueprint_name = 'pages_blueprint'

blueprint = Blueprint(
    blueprint_name,
    __name__,
    url_prefix=''
)

@blueprint.route('/')
def index():
    return render_template('pages/index.html', segment='index') #,

@blueprint.route('/preditor_prey_game')
def pred_prey_game():
    return render_template('pages/preditor_prey_game.html', segment='index') #,



@blueprint.route('/<template>')
# @login_required
def route_template(template):
    try:
        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("pages/" + template,segment=segment) #,

    except TemplateNotFound:
        return render_template('pages/page-404.html'), 404

    except:
        return render_template('pages/page-500.html'), 500

# Helper - Extract current page name from request
def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        segment = segment.path.split('.')[-1]
        print(segment)
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None
