from dominate.tags import img
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import *


def add_nav(app):
    logo = img(src='./static/img/logo.jpg', height="50", width="50", style="margin-top:-15px")
    topbar = Navbar(logo,
                    View('Home', f'index'),
                    Subgroup('Participate',
                        View('Information','participate'),
                        Separator(),  Text('Experiments'),
                        View('Preditor-Prey Game', 'render_PursuitGame'),
                        View('Block Game', 'render_iRobot'),
                    ),
                    Subgroup('About Us',
                             View('Researchers', 'researchers'),
                             View('Research Goals',  'research_goals'),
                             View('Contact',  'contact'),
                             ),
                    )

    # # registers the "top" menubar
    nav = Nav()
    nav.register_element('top', topbar)

    Bootstrap(app)
    nav.init_app(app)
    return nav
