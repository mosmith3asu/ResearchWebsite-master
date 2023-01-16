
from flask_nav import Nav
from flask_nav.elements import *
from dominate.tags import img
# from .routes import route_template
def init_nav():
    bp_route = 'pages_blueprint' +'.' #+ 'route_template'
    logo = img(src='./static/img/logo.jpg', height="50", width="50", style="margin-top:-15px")
    topbar = Navbar(logo,
                    View('Home', bp_route +  f'index'),
                    Subgroup('Participate',
                        View('Information',bp_route + 'participate'),
                        Separator(),  Text('Experiments'),
                        View('Preditor-Prey Game', bp_route + 'render_PursuitGame'),
                        View('Block Game', bp_route + 'render_iRobot'),
                    ),
                    Subgroup('About Us',
                             View('Researchers', bp_route + 'researchers'),
                             View('Research Goals',  bp_route + 'research_goals'),
                             View('Contact',  bp_route + 'contact'),
                             ),
                    )

    # logo = img(src='./static/img/logo.jpg', height="50", width="50", style="margin-top:-15px")
    # topbar = Navbar(logo,
    #                 View('Home', f'index'),
    #                 Subgroup('Participate',
    #                     View('Information','participate'),
    #                     Separator(),  Text('Experiments'),
    #                     View('Preditor-Prey Game', 'render_PursuitGame'),
    #                     View('Block Game', 'render_iRobot'),
    #                 ),
    #                 Subgroup('About Us',
    #                          View('Researchers', 'researchers'),
    #                          View('Research Goals',  'research_goals'),
    #                          View('Contact',  'contact'),
    #                          ),
    #                 )


    # # registers the "top" menubar
    nav = Nav()
    nav.register_element('top', topbar)
    return nav
