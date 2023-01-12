
from flask_nav import Nav
from flask_nav.elements import *
from dominate.tags import img
# from .routes import route_template
def init_nav():
    bp_route = 'pages_blueprint' +'.' + 'route_template'
    logo = img(src='./static/img/logo.jpg', height="50", width="50", style="margin-top:-15px")
    topbar = Navbar(logo,
                    View('Home', bp_route,template= f'index'),
                    Subgroup('Participate',
                        View('Information',bp_route,template='participate'),
                        Separator(),  Text('Experiments'),
                        View('Preditor-Prey Game', bp_route,template='preditor_prey_game'),
                        View('Block Game', bp_route,template='block_game'),
                    ),
                    Subgroup('About Us',
                             View('Researchers', bp_route,template='researchers'),
                             View('Research Goals',  bp_route,template='research_goals'),
                             View('Contact',  bp_route,template='preditor_prey_game'),
                             ),
                    )


    # # registers the "top" menubar
    nav = Nav()
    nav.register_element('top', topbar)
    return nav
