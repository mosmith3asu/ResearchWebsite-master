from flask import request

from apps import session
from apps.static.PursuitGame.page_assets import test_views


def add_events(socketio):
    @socketio.on('connect')
    def event_connect():
        session['sid'] = request.sid
        print(f'Client Connected [sid: {request.sid}]...')

    @socketio.on('update_gamestate')
    def event_update_gamestate(message):

        send_data = {}
        GAME = session.get("GAME")
        iview = session.get("iview")
        if 'keypress' in message.keys():
            GAME.sample_user_input(message['keypress'])
            # print(message['keypress'])
        if 'button' in message.keys():
            if message['button'] == 'continue':

                # ADDED #################################
                if 'canvas' in test_views[iview-1]['view']:
                    print('Next game')
                    GAME.is_finished = True
                    GAME.playing = False
                #########################################


                iview += 1
            elif message['button'] == 'back':
                iview -= 1
        if 'submit_survey' in message.keys():
            iview += 1
            responses = message['submit_survey']
            print(f"SURVEY: {responses}")
        if 'submit_background' in message.keys():
            iview += 1
            responses = message['submit_background']
            print(f"BACKGROUND: {responses}")

        if 'canvas' in test_views[iview]['view']:
            if GAME.is_finished:
                print(f'RESTARTING GAME')
                GAME.new_world()
                GAME.is_finished = False
                GAME.playing = True
            else:
                GAME.tick()
            send_data = GAME.get_gamestate()

            # print(send_data)

        for key in test_views[iview].keys():
            send_data[key] = test_views[iview][key]


        session['iview'] = iview
        # print(f'Send {iview}')
        # RESPOND TO THE CLIENT --------------------
        # print(f'Host Rec: {message} Host Send: {send_data}')
        socketio.emit('update_gamestate', send_data, room=session['sid'])  #

    # socketio.on_event(message='connect', handler=event_connect)
    # socketio.on_event(message='update_gamestate', handler=event_update_gamestate)

