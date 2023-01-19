
def initialize(app,socketio):
    from .routes import add_routes
    from .navigation import add_nav
    from .events import add_events
    add_nav(app)
    add_routes(app)
    add_events(socketio)
