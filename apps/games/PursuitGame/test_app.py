import copy
import numpy as np
from flask import Flask, render_template,session,request
from flask_session import Session
from flask_socketio import SocketIO, emit
from uuid import uuid4
# from Web_Server.game_backend.game_handler import JointPursuitGame
import time
from datetime import datetime



DEBUG = True


app = Flask(__name__)
APP_KEY = '92659'
app.config['SECRET_KEY'] = APP_KEY
# configure the session, making it permanent and setting the session type to the filesystem.
# app.config["SESSION_PERMANENT"] = True #set the session to permanent. This means that the session cookies won’t expire when the browser closes.
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem" # We set the session type to the filesystem, which means that the cookies are going to be stored locally on the server-side.
app.config['SECRET_KEY'] = uuid4().hex # Configure secret key for encryption
app.config["TEMPLATES_AUTO_RELOAD"] = True
# app.config["EXPLAIN_TEMPLATE_LOADING"] = True
Session(app)
app.debug = True
socketio = SocketIO(app)
# GAME = JointPursuitGame()
# GAME.DEBUG = DEBUG

# if DEBUG:
#     GAME.MASTER_STAGE = 0
#     GAME.pretrial_stage = 0


@socketio.on('connect')
def connect():
    """
    store in the session the user socket ID sid. We store it when the user first connects to the page using.
    When the connect() event happens, we store the user socket ID on the session variable sid.
    """
    session['sid'] = request.sid

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('script_test.html')


@socketio.on('next_pos')
def next_pos(message):
    VERBOSE = False


    if GAME.playing_game:
        if message['action']   == 'left':   GAME.execute['last_action'] = [-1 , 0]
        elif message['action'] == 'right':  GAME.execute['last_action'] = [ 1 , 0]
        elif message['action'] == 'up':     GAME.execute['last_action'] = [ 0, -1]
        elif message['action'] == 'down':   GAME.execute['last_action'] = [ 0,  1]

        # if VERBOSE:
        #     print(f'ACTOiM DETECTED: ')
        #     print(f'\t| pos0 = {GAME.pos}')

        # if message['action']   == 'left':   pos = GAME.move_player([-1 , 0])
        # elif message['action'] == 'right':  pos = GAME.move_player([ 1 , 0])
        # elif message['action'] == 'up':     pos = GAME.move_player([ 0, -1])
        # elif message['action'] == 'down':   pos = GAME.move_player([ 0,  1])
        # else: pos = GAME.move_player([0, 0])
        # GAME.pos[1] = GAME.pos[2] #<============= DEBUGGING =========================-
        # GAME.player_pen = -5 #<============= DEBUGGING =========================-

        # RESPOND TO THE CLIENT --------------------
        # Emit the updated position back to canvas.js
        # if : socketio.emit('game_running', {"running": 1}, room=session['sid'])
        # data = {}
        # data['pos'] = pos
        # data['timer'] = GAME.get_timer()
        # data['moves'] = GAME.moves
        # data['player_pen'] = GAME.player_pen
        # data['received_penalty'] = GAME.check_player_in_penalty()
        # data['finished'] = GAME.check_finished()
        #
        # if VERBOSE:
        #     for key in data: print(f'\t| {key} = {data[key]}')
        #     print(f'\t| execute = {GAME.execute["enable"]}')
        # socketio.emit('update_status',data, room=session['sid'])
        # socketio.emit('update_status', data)



##########################################################
# UPDATE TIMER FREQUENTLY ################################
def push_timer():
    # global GAME
    i = 0

    delay = 0.1
    while True:
        try:
            # if GAME.playing_game:
                # GAME.tictock()
                with app.test_request_context('/'):
                    # if not GAME.check_finished() :
                    #     GAME.move_partner()
                    #     GAME.move_player()
                    #     GAME.move_evader()


                    # print(f'\r Moves({GAME.execute["enable"]}) = {GAME.moves} \t A={GAME.execute["last_action"]}',end='')
                    data = {}
                    data['iworld'] = 1
                    data['state'] = [1,1,3,3,5,5]
                    data['timer'] = 0.25
                    data['pen_alpha'] = 0.0
                    data['nPen'] = i
                    data['moves'] = 18
                    data['playing'] = True
                    data['is_finished'] = False
                    data['penalty_states'] = [[1,1]]

                    # data['pos'] = GAME.pos
                    # data['last_pos'] = GAME.last_pos
                    # data['timer'] = GAME.get_timer()
                    # data['moves'] = GAME.moves
                    # data['player_pen'] = GAME.player_pen
                    # # data['received_penalty'] = GAME.check_player_in_penalty()
                    # data['pen_alpha'] = GAME.check_player_in_penalty()
                    # data['finished'] = GAME.check_finished()
                    # data['close_game'] = GAME.close_game

                    # curr_action = GAME.execute['last_action']
                    # if   np.all(curr_action == [ 1, 0]): last_input = 'right'
                    # elif np.all(curr_action == [-1, 0]): last_input = 'left'
                    # elif np.all(curr_action == [0, 1]):  last_input = 'down'
                    # elif np.all(curr_action == [0, -1]): last_input = 'up'
                    # else: last_input = 'wait'
                    # data['last_input'] = last_input

                    # print(data['pen_alpha'])
                    socketio.emit('update_gamestate', data)
                    i += 1
            # else:
            #     print(f'\r Game off', end='')

        except Exception as exp:
            print(exp)
            print(f'\r Push Timer Error',end='')
            # print('push_timer: err')
        socketio.sleep(delay)









##########################################################
# RUN APP ################################################
if __name__ == "__main__":
    # app.run(debug=True)
    socketio.start_background_task(target=push_timer)
    socketio.run(app,host="192.168.0.137",port=8080, debug=True,use_reloader=False)
