# from datetime import datetime
import copy
import logging
import time
import numpy as np
import math
class GameHandler(object):
    def __init__(self,iworld,debug = True):
        self.debug = debug

        print('INITIALIZING GAME')
        self.iworld = iworld
        self.done = False # game is done and disable move
        self.is_finished = False # ready to advance to next pate

        self.state = [1,1,3,3,5,5]
        self.penalty_states = [[1,1],[1,2],]


        self.penalty_counter = 0
        self.remaining_moves = 20 if not self.debug else 3
        self.move_enables = {}
        self.move_enables['R'] = True
        self.move_enables['H'] = True
        self.move_enables['E'] = True


        self.t_evader_move_delay = 0.5
        self.t_finished_move_delay = 0.5
        self.t_finished_overlay_delay = 3
        self.t_move_dur = 3.0 if not self.debug else 1.0
        self.t_pen_overlay_dur = (2/3) * self.t_move_dur
        self.t_last_pen = time.time()
        self.t_move_start = time.time()


        self.t_start = time.time()
        self.timer = time.time() - self.t_start
        self.timer_max_val = 1
        self.timer_min_val = 0

        self.pen_max_alpha = 1
        self.pen_alpha = self.pen_max_alpha

        self.a2name = {}
        self.a2idx = {}
        self.a2move = {}
        self.a2move['down'] = [1,0]
        self.a2move['left'] = [0,-1]
        self.a2move['up'] = [-1,0]
        self.a2move['right'] = [0,1]
        self.a2move['wait'] = [0,0]
        self.a2move['spacebar'] = [0, 0]

        self.a2idx['down'] = 0  # self.a2move[0] = [1,0]
        self.a2idx['left'] = 1  # self.a2move[1] = [0,-1]
        self.a2idx['up'] = 2  # self.a2move[2] = [-1,0]
        self.a2idx['right'] = 3  # self.a2move[3] = [0,1]
        self.a2idx['wait'] = 4  # self.a2move[4] = [0,0]
        self.a2idx['spacebar'] = 4

        for aname in ['down','left','up','right','spacebar','wait']:
            self.a2name[self.a2idx[aname]] = aname
            self.a2name[tuple(self.a2move[aname])] = aname
            # self.a2move[aname] = self.a2move[aname]
            self.a2move[self.a2idx[aname]] = self.a2move[aname]
            # self.a2idx[aname] = self.a2idx[aname]
            self.a2idx[tuple(self.a2move[aname])] = aname

        self.current_action_idx = {}
        self.current_action_idx['H'] = self.a2idx['wait']
        self.current_action_idx['R'] = self.a2idx['wait']
        self.current_action_idx['E'] = self.a2idx['up']

        self.default_settings = {}
        for key in self.__dict__.keys():
            self.default_settings[key] = copy.deepcopy(self.__dict__[key])
    def tick(self,verbose = False):
        move_duration = time.time()-self.t_move_start
        post_move_duration = -1*min([0,self.t_move_dur - move_duration])

        if not self.done:
            perc_complete = (time.time() - self.t_last_pen)/self.t_pen_overlay_dur
            self.pen_alpha =  self.pen_max_alpha * max([0,1-perc_complete])

            self.timer = move_duration/self.t_move_dur
            self.timer = min([self.timer_max_val, self.timer])
            self.timer = max([self.timer_min_val, self.timer])
            executing = (self.timer == self.timer_max_val)

            if executing:
                if verbose: print(f'EXECUTING:{self.a2name[self.current_action_idx["H"]]}')
                players_finished = self.execute_players()
                evader_finished = self.execute_evader(post_move_duration)
                if players_finished and evader_finished:
                    self.new_move(post_move_duration)

            # Check Closing Gamestate
            self.done = self.check_done()
        self.is_finished = self.close_world(post_move_duration)


    def close_world(self,t_post_move):
        finished = False
        if self.done:
            # self.is_finished = True
            if t_post_move >= self.t_finished_overlay_delay:
                finished= True
        return finished

    def check_done(self):
        dist_R2E = math.dist(self.state[0:2],self.state[4:6])
        dist_H2E = math.dist(self.state[2:4],self.state[4:6])
        is_caught = (dist_R2E <=1 and dist_H2E <=1)
        no_remaining_moves = (self.remaining_moves <= 0)
        if is_caught or no_remaining_moves:
            done = True
            self.move_enables['R'] = False
            self.move_enables['H'] = False
            self.move_enables['E'] = False
        else:  done = False
        return done

    def new_world(self,iworld):
        self.__init__(iworld)
        # for key in self.default_settings.keys():
        #     self.__dict__[key] = copy.deepcopy(self.default_settings[key])
        # self.iworld = iworld
    def new_move(self,t_post_move):
        is_last_move = (self.remaining_moves <= 1)

        total_delay = self.t_finished_move_delay
        total_delay += self.t_evader_move_delay if not is_last_move else 0
        if t_post_move >= total_delay:
            self.t_move_start = time.time()
            self.remaining_moves = max(0,self.remaining_moves - 1)

            self.move_enables['R'] = True
            self.move_enables['H'] = True
            self.move_enables['E'] = True
    def execute_players(self):
        move_R = self.a2move[self.current_action_idx['R']]
        move_H = self.a2move[self.current_action_idx['H']]
        move_E = self.a2move['wait']

        move_R = move_R if self.move_enables['R'] else self.a2move['wait']
        move_H = move_H if self.move_enables['H'] else self.a2move['wait']
        move_E = move_E if self.move_enables['E'] else self.a2move['wait']
        new_state = np.array(self.state) + np.array(move_R + move_H + move_E)

        self.state = [int(s) for s in new_state]
        finished = True

        if finished:
            if self.move_enables['H']: # only write no-action once
                self.current_action_idx['H'] = self.a2idx['wait']
            self.move_enables['R'] = False
            self.move_enables['H'] = False
        return finished

    def execute_evader(self,t_post_move):
        is_last_move = (self.remaining_moves <= 1)
        TIME2MOVE = (t_post_move >= self.t_evader_move_delay)
        move_R = self.a2move['wait']
        move_H = self.a2move['wait']
        move_E = self.a2move['wait']

        finished = False
        if TIME2MOVE:
            move_E = self.a2move[self.current_action_idx['E']]
            finished = True

        move_R = move_R if self.move_enables['R'] else self.a2move['wait']
        move_H = move_H if self.move_enables['H'] else self.a2move['wait']
        move_E = move_E if self.move_enables['E'] else self.a2move['wait']
        move_E = move_E if not is_last_move else self.a2move['wait']

        new_state = np.array(self.state) + np.array(move_R + move_H + move_E)
        self.state = [int(s) for s in new_state]

        if finished:
            self.move_enables['E'] = False

        return finished
    def sample_user_input(self,key_input,verbose = False):
        if key_input == 'None':
            new_action = self.current_action_idx['H']
        elif key_input in self.a2idx.keys():
            new_action = self.a2idx[key_input]
            if verbose: print(f'KEY INPUT:{key_input}=>{new_action}')
        else:
            new_action = self.current_action_idx['H']
            logging.warning(f'User input unknown: {key_input}')
        self.current_action_idx['H'] = new_action

    def get_gamestate(self):
        data = {}
        data['iworld'] = self.iworld
        data['state'] = self.state
        data['timer'] = self.timer
        data['pen_alpha'] = self.pen_alpha
        data['nPen'] = self.penalty_counter
        data['moves'] = self.remaining_moves
        data['playing'] = self.done
        data['is_finished'] = self.is_finished
        data['penalty_states'] = self.penalty_states
        data['current_action'] = self.current_action_idx['H']
        return data