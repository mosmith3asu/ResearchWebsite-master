import os
from datetime import datetime
class PursuitGame_PageHandler(object):
    def __init__(self,GAME,start_stage = None):
        print('INITIALIZING PAGES')

        self.treatment = {'pen':-3,'prob':0.5}
        self.nGames = 7

        self.GAME = GAME
        self.stage = 0 #if start_stage is not None else start_stage
        self.iworld = -1
        self.request = None
        # module_root ='/'.join(os.getcwd().split('\\')[:-2]) + '/'
        # template_dir = module_root + 'templates/'
        # self.template = 'script_test.html'
        # self.kwargs = {}

        # self.ext = r"C:\Users\mason\Desktop\ResearchWebsite-master\apps\static\PursuitGame\templates" +"\\"
        self.ext = r"PursuitGame/"

        i = 0
        self.trial_templates = {}
        # PRETRIAL #######################
        self.trial_templates[i] = ['image_button_page.html', {'file': 'Page_Consent.jpg'}]; i += 1
        self.trial_templates[i] = ['background.html', {}]; i += 1
        self.trial_templates[i] = ['image_button_page.html', {'file': 'Page_Exclusion.jpg'}]; i += 1
        self.trial_templates[i] = ['survey_info.html', {}]; i += 1
        self.trial_templates[i] = ['survey_pretrial.html', {}]; i += 1
        self.trial_templates[i] = ['instructions_movement.html', {}]; i += 1
        self.trial_templates[i] = ['instructions_turns.html', {}]; i += 1
        self.trial_templates[i] = ['instructions_penalty.html', {}]; i += 1
        self.trial_templates[i] = ['instructions_objective.html', {}]; i += 1
        self.trial_templates[i] = ['instructions_practice.html', {}]; i += 1
        self.trial_templates[i] = ['game_canvas.html', {}]; i += 1
        self.trial_templates[i] = ['instructions_end.html', {}]; i += 1


        # TRIAL #####################################
        for igame in range(self.nGames):
            self.trial_templates[i] = ['risk_information.html', self.treatment];i += 1
            self.trial_templates[i] = ['game_canvas.html', {}];i += 1
            self.trial_templates[i] = ['survey.html', {}];i += 1


        # POST-TRIAL DEBRIEF #################################

        self.trial_templates[i] = ['debrief_main.html', {}]; i += 1
        self.trial_templates[i] = ['debrief_info.html', {}]; i += 1


        self.current_template,self.current_kwargs = self.trial_templates[0]




    def unpack_survey(self,request):
        try:
            info = {}
            for r in range(7):  info[r] = request.form[f'row-{r+1}']
            if None in info.values(): info = None
        except:  info = None
        return info
    ###########################################
    ###### PRETRIAL ##########################
    def check_inclusion_criteria(self,info):
        return True
    def unpack_background_survey(self,request):
        try:
            info = {}
            now = datetime.now()
            info['date'] = now.strftime("%m/%d/%Y")
            info['time'] = now.strftime("%H:%M:%S")
            info['age'] = request.form[f"age"]
            info['sex'] = request.form[f"sex"]
            info['occupation'] = request.form[f"occupation"]
            info['experience'] = request.form[f"experience"]
            info['R_pen'] = 0  # treatment['pen']
            info['P_pen'] = 0 #treatment['prob']
            info['sid'] = 'no sid'  # self.request.sid
            if None in info.values(): info = None
        except:  info = None
        return info
    def run_pretrial(self,request):
        template = self.current_template
        kwargs = self.current_kwargs

        # Instructions next button ---------------------------------------
        if request.form.get("submit_back"):
            self.stage -= 1; print(f'PRETRIAL: back button pressed => stage = {self.stage}')
            template, kwargs = self.trial_templates[self.stage]
            print(f'PRETRIAL: back button pressed => stage[{self.stage}]: {template}')

        elif request.form.get("submit_continue"):
            self.stage += 1
            template, kwargs = self.trial_templates[self.stage]
            print(f'PRETRIAL: continue button pressed => stage[{self.stage}]: {template}')


        # Background info submit button ----------------------------------
        elif request.form.get("submit_background_response"):
            info = self.unpack_background_survey(request)
            if info is None:
                print(f'Incomplete survey')
                template = 'background.html'
                kwargs = {'msg':  '*Make sure to respond to each question before continuing'}
            else:
                if self.check_inclusion_criteria(info):
                    self.stage += 1
                    template, kwargs = self.trial_templates[self.stage]
                else:
                    template = 'image_button_page.html'
                    kwargs = {'file': 'Page_Exclusion.jpg'}

        # Baseline Survey ----------------------------------
        elif request.form.get("submit_survey_response"):
            info = self.unpack_survey(request)
            if info is None:
                print(f'Incomplete survey')
                template, _ = self.trial_templates[self.stage]
                kwargs = {'msg':  '*Make sure to respond to each question before continuing'}
            else:
                self.stage += 1
                template, kwargs = self.trial_templates[self.stage]


        # Practice Game ----------------------------------------------------
        elif request.form.get("submit_practice"):
            self.GAME.new_world(iworld=0)
            self.stage += 1
            template, kwargs = self.trial_templates[self.stage]

        elif request.form.get("advance"): # end of game post sent by client
            print(f'GAME FINISHED POST')
            # GAME.playing_game = False
            self.stage += 1
            template, kwargs = self.trial_templates[self.stage]

        return template,kwargs
    def get_page(self,request):
        template, kwargs = self.run_pretrial(request)
        self.current_template = template
        self.current_kwargs = kwargs

        return  self.ext+template,kwargs

