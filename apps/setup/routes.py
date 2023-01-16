from flask import Blueprint
from flask import render_template, request
# from flask_login import login_required
from jinja2 import TemplateNotFound
# from .. import app
from . import blueprint
from .. import session
from ..static.PursuitGame.static.scripts.game_handler import GameHandler
from ..static.PursuitGame.static.scripts.game_page_handler import PursuitGame_PageHandler

# blueprint_name = 'main'
# blueprint = Blueprint(  blueprint_name, __name__,  url_prefix='')
######################################################################################
@blueprint.route('/')
def index():
    try: return render_template('pages/index.html', segment='index') #,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except: return render_template('pages/page-500.html'), 500

######################################################################################
route_name = 'about'
@blueprint.route(f'/{route_name}')
def about():
    try: return render_template(f'pages/{route_name}.html', segment='index') #,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except: return render_template('pages/page-500.html'), 500

######################################################################################
route_name = 'contact'
@blueprint.route(f'/contact')
def contact():
    try:  return render_template(f'pages/contact.html')  # ,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except: return render_template('pages/page-500.html'), 500

######################################################################################
route_name = 'index'
@blueprint.route(f'/{route_name}')
def index2():
    try: return render_template(f'pages/{route_name}.html', segment='index') #,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except: return render_template('pages/page-500.html'), 500

######################################################################################
route_name = 'participate'
@blueprint.route(f'/{route_name}')
def participate():
    try:  return render_template(f'pages/{route_name}.html', segment='index')  # ,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except:  return render_template('pages/page-500.html'), 500

######################################################################################
route_name = 'render_iRobot'
@blueprint.route(f'/{route_name}')
def render_iRobot():
    try: return render_template(f'pages/{route_name}.html', segment='index') #,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except: return render_template('pages/page-500.html'), 500


######################################################################################
@blueprint.route(f'/render_PursuitGame',methods=['GET', 'POST'])
def render_PursuitGame():
    if not session.get('GAME'):
        session['GAME'] = GameHandler(iworld=1)
    if not session.get('PAGES'):
        # start_stage = DEBUG_STAGE if DEBUG else None
        # session['PAGES'] = PursuitGame_PageHandler(session['GAME'],start_stage=start_stage)
        session['PAGES'] = PursuitGame_PageHandler(session['GAME'])

    PAGES = session.get('PAGES')
    template, kwargs = PAGES.get_page(request)
    # print(session)
    print(f'[{PAGES.stage}]', template, kwargs)
    return render_template(template, **kwargs)



    # return render_template(f'pages/render_PursuitGame.html')  # ,
    # try:
    # except TemplateNotFound: return render_template('pages/page-404.html'), 404
    # except: return render_template('pages/page-500.html'), 500


######################################################################################
route_name = 'research_goals'
@blueprint.route(f'/{route_name}')
def research_goals():
    try: return render_template(f'pages/{route_name}.html') #,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except: return render_template('pages/page-500.html'), 500


######################################################################################
route_name = 'researchers'
@blueprint.route(f'/{route_name}')
def researchers():
    try: return render_template(f'pages/{route_name}.html') #,
    except TemplateNotFound: return render_template('pages/page-404.html'), 404
    except: return render_template('pages/page-500.html'), 500

