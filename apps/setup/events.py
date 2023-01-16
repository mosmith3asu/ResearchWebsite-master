# import numpy as np
import logging
from flask import session,request
from uuid import uuid4
from .. import socketio #app, DEBUG
from ..static.PursuitGame.static.scripts.game_handler import GameHandler
from ..static.PursuitGame.static.scripts.game_page_handler  import PursuitGame_PageHandler




@socketio.on('connect')
def connect():
    """
    store in the session the user socket ID sid. We store it when the user first connects to the page using.
    When the connect() event happens, we store the user socket ID on the session variable sid.
    """
    session['sid'] = request.sid
    # session['GAME'] = GameHandler(iworld=1)
    # session['PAGES'] = PursuitGame_PageHandler(session['GAME'])
    # socketio.start_background_task(target=update_gamestate,room=session['sid'])
    # socketio.emit('update_gamestate', GAME.get_gamestate(), room=session['sid'])  #
    print(session)

@socketio.on('update_gamestate')
def update_gamestate(message):
    GAME = session.get("GAME")
    # PAGES = session.get("PAGES")
    # GAME = PAGES.GAME
    if 'keypress' in message.keys():
        GAME.sample_user_input(message['keypress'])
    else:
        logging.warning(f'unknown client message: {message}')
    GAME.tick()

    # RESPOND TO THE CLIENT --------------------
    socketio.emit('update_gamestate', GAME.get_gamestate(), room=session['sid'])  #
#
# @app.route("/", methods=['GET', 'POST'])
# def home():
#     if not session.get('GAME'):
#         session['GAME'] = GameHandler(iworld=1)
#     if not session.get('PAGES'):
#         # start_stage = DEBUG_STAGE if DEBUG else None
#         # session['PAGES'] = PursuitGame_PageHandler(session['GAME'],start_stage=start_stage)
#         session['PAGES'] = PursuitGame_PageHandler(session['GAME'])
#
#     PAGES = session.get('PAGES')
#     template, kwargs = PAGES.get_page(request)
#     # print(session)
#     print(f'[{PAGES.stage}]', template, kwargs)
#     return render_template(template, **kwargs)
#     # return render_template('script_test.html')
# return app,
