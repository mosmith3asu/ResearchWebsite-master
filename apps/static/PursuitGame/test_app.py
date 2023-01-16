import logging
import os

from flask import Flask, render_template,session,request,redirect
from flask_session import Session
from flask_socketio import SocketIO
from uuid import uuid4
from static.scripts.game_handler import GameHandler
from static.scripts.game_page_handler import PursuitGame_PageHandler

DEBUG = True
DEBUG_STAGE = 3

APP_KEY = '92659'



app = Flask(__name__,
            static_url_path='',
            static_folder='C:\\Users\\mason\\Desktop\\ResearchWebsite-master\\apps\\static',
            template_folder='C:\\Users\\mason\\Desktop\\ResearchWebsite-master\\apps\\templates')
app.debug = DEBUG
app.config['SECRET_KEY'] = APP_KEY
# app.config["SESSION_PERMANENT"] = True #set the session to permanent. This means that the session cookies won’t expire when the browser closes.
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem" # We set the session type to the filesystem, which means that the cookies are going to be stored locally on the server-side.
app.config['SECRET_KEY'] = uuid4().hex # Configure secret key for encryption
app.config["TEMPLATES_AUTO_RELOAD"] = True
# app.config["EXPLAIN_TEMPLATE_LOADING"] = True


Session(app)
socketio = SocketIO(app)


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
    if 'keypress' in message.keys(): GAME.sample_user_input(message['keypress'])
    else: logging.warning(f'unknown client message: {message}')
    GAME.tick()

    # RESPOND TO THE CLIENT --------------------
    socketio.emit('update_gamestate', GAME.get_gamestate(), room=session['sid'])  #


@app.route("/", methods=['GET', 'POST'])
def home():
    if not session.get('GAME'):
        session['GAME'] = GameHandler(iworld=1)
    if not session.get('PAGES'):
        # start_stage = DEBUG_STAGE if DEBUG else None
        # session['PAGES'] = PursuitGame_PageHandler(session['GAME'],start_stage=start_stage)
        session['PAGES'] = PursuitGame_PageHandler(session['GAME'])


    PAGES = session.get('PAGES')
    template, kwargs = PAGES.get_page(request)
    # print(session)
    print(f'[{PAGES.stage}]',template,kwargs)
    # return render_template('/static/PursuitGame/templates/index.html', **kwargs)
    return render_template(template, **kwargs)
    # return render_template('script_test.html')



def remove_session_data():
    logging.warning('REMOVING FLASK SESSION DATA')
    root_dir = os.getcwd()#.split('\\')
    content_dir = root_dir+'\\flask_session\\'
    for fname in os.listdir(content_dir):
        os.remove(content_dir + fname)
##########################################################
# RUN APP ################################################
if __name__ == "__main__":
    # if DEBUG:
    #     remove_session_data()
    # app.run(debug=True)
    # socketio.start_background_task(target=push_timer)
    socketio.run(app,host="192.168.0.137",port=8080, debug=True,use_reloader=False)
