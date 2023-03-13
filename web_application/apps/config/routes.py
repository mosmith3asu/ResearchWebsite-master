from flask import render_template
from jinja2 import TemplateNotFound
from .. import session
from apps.static.PursuitGame.game_handler import GameHandler


def add_routes(app):

    def index():
        try: return render_template('pages/index.html', segment='index') #,
        except TemplateNotFound: return render_template('pages/page-404.html'), 404
        except: return render_template('pages/page-500.html'), 500

    def about():
        try: return render_template(f'pages/about.html', segment='index') #,
        except TemplateNotFound: return render_template('pages/page-404.html'), 404
        except: return render_template('pages/page-500.html'), 500

    def contact():
        try:  return render_template(f'pages/contact.html')  # ,
        except TemplateNotFound: return render_template('pages/page-404.html'), 404
        except: return render_template('pages/page-500.html'), 500

    def participate():
        try:  return render_template(f'pages/participate.html', segment='index')  # ,
        except TemplateNotFound: return render_template('pages/page-404.html'), 404
        except:  return render_template('pages/page-500.html'), 500

    def render_iRobot():
        try: return render_template(f'pages/render_iRobot.html', segment='index') #,
        except TemplateNotFound: return render_template('pages/page-404.html'), 404
        except: return render_template('pages/page-500.html'), 500

    def render_PursuitGame():
        if not session.get('iview'):
            session['iview'] = 0
        if not session.get('GAME'):
            # treatment = GameHandler.sample_treatment()
            # session['GAME'] = GameHandler(iworld=0,treatment=treatment)
            session['GAME'] = GameHandler.new()
        return render_template('pages/render_PursuitGame.html')

        # if not session.get('GAME'):
        #     session['GAME'] = GameHandler(iworld=1)
        # if not session.get('PAGES'):
        #     # start_stage = DEBUG_STAGE if DEBUG else None
        #     # session['PAGES'] = PursuitGame_PageHandler(session['GAME'],start_stage=start_stage)
        #     session['PAGES'] = PursuitGame_PageHandler(session['GAME'])
        #
        # PAGES = session.get('PAGES')
        # template, kwargs = PAGES.get_page(request)
        # # print(session)
        # print(f'[{PAGES.stage}]', template, kwargs)
        # return render_template(template, **kwargs)

    def research_goals():
        try: return render_template(f'pages/research_goals.html') #,
        except TemplateNotFound: return render_template('pages/page-404.html'), 404
        except: return render_template('pages/page-500.html'), 500

    def researchers():
        try: return render_template(f'pages/researchers.html') #,
        except TemplateNotFound: return render_template('pages/page-404.html'), 404
        except: return render_template('pages/page-500.html'), 500


    routes = {}
    routes['/'] = index
    routes['/index'] = index
    routes['/home'] = index

    routes['/contact'] = contact
    routes['/participate'] = participate
    routes['/render_iRobot'] = render_iRobot
    routes['/render_PursuitGame'] = render_PursuitGame
    routes['/research_goals'] = research_goals
    routes['/researchers'] = researchers
    routes['/about'] = about

    for endpoint in routes.keys():
        app.add_url_rule(endpoint, view_func=routes[endpoint], methods=['GET', 'POST'])

