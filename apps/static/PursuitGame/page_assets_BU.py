



GithubImgs = {}
GithubImgs['consent'] = 'https://github.com/mosmith3asu/ResearchWebsite-master/blob/master/apps/static/PursuitGame/static/IMG/Page_Consent.jpg?raw=true'
GithubImgs['controls'] = 'https://github.com/mosmith3asu/ResearchWebsite-master/blob/master/apps/static/PursuitGame/static/IMG/Page_Controls.jpg?raw=true'
GithubImgs['movement'] = "https://github.com/mosmith3asu/ResearchWebsite-master/blob/master/apps/static/PursuitGame/static/IMG/Instructions_Movement.gif?raw=true"
GithubImgs['turns'] =  "https://github.com/mosmith3asu/ResearchWebsite-master/blob/master/apps/static/PursuitGame/static/IMG/Instructions_Turns.gif?raw=true"
GithubImgs['penalties'] =  "https://raw.githubusercontent.com/mosmith3asu/ResearchWebsite-master/master/apps/static/PursuitGame/static/IMG/Instructions_Penalty.gif"
GithubImgs['objective'] =  "https://github.com/mosmith3asu/ResearchWebsite-master/blob/master/apps/static/PursuitGame/static/IMG/Instructions_Objective.gif?raw=true"
# GithubImgs['objective'] =  "https://github.com/mosmith3asu/ResearchWebsite-master/blob/master/apps/static/PursuitGame/static/IMG/Instructions_Objective.gif?raw=true"


TEXT = {}

TEXT['consent'] = "I am a graduate student under the direction of Professor Wenlong Zhang in the Ira Fulton Schools of Engineering at Arizona State University.  I am conducting a research study to investigate human interaction and decision-making under risky situations. <br><br>" \
"Your participation will involve playing a series of seven games with a virtual partner taking an estimated total of 15 minutes for the whole experiment. The game will be played in a web interface and will require both a mouse and computer to play. You have the right not to answer any question, and to stop participation at any time.<br><br>" \
"Your participation in this study is voluntary.  If you choose not to participate or to withdraw from the study at any time, there will be no penalty. You must be 18 or older, able to operate a computer, and meet the required demographic group inquired before the experiment begins to participate in this study.<br><br>" \
"Your participation will benefit research in human decision-making for the purpose of improving control models for robots that interact with humans in dynamic ways. Specifically, your responses to surveys will help investigate how human trust in virtual agents developed under different models of behavior. Also, the actions you take during the game will help validate and improve robotic understanding of humans. After the experiment is complete, further explanation about how the virtual partner modeled your actions is available if you wish to learn more about how you are contributing to this field. There are no foreseeable risks or discomforts to your participation.<br><br>" \
"Responses to all survey questions, participation date, and the actions you take during the game will be collected. To protect your privacy, all personal identifiers will be removed from your data and stored on a secure server. Your data will be confidential and the de-identified data collected as a part of this study will not be shared with others outside of the research team approved by this study’s IRB or be used for future research purposes or other uses. The results of this study may be used in reports, presentations, or publications but your name will not be used. <br><br>" \
"If you have any questions concerning the research study, please contact the research team at: mosmith3@asu.edu or wenlong.zhang@asu.edu. If you have any questions about your rights as a subject/participant in this research, or if you feel you have been placed at risk, you can contact the Chair of the Human Subjects Institutional Review Board, through the ASU Office of Research Integrity and Assurance, at (480) 965-6788. Please let me know if you wish to be part of the study.<br><br>" \



TEXT['survey info'] = '<ul>' \
'<p class="instructions_head">Please complete the survey on the following page:</p>' \
'<li>Questions relate to how you generally feel when interacting with artificial agents </li>' \
'<li>Awnser the questions to the best of your ability</li>' \
'<li>Click the bubble to record your response</li>' \
'<li>When you are finished, submit the survey to continue to the experiment instructions</li>' \
'</ul>'

TEXT['movement'] = '<ul><p class="instructions_head">Moving Your Character:</p>' \
'<li>Your character is the <span style="color: #00008b;font-weight:bold;">dark blue square</span> and your partner is the <span style="color: #89CFF0;font-weight:bold;">smaller light blue square</span></li>' \
'<li>You can input the up, down, left, or right action by pressing the arrow keys on the keyboard </li>' \
'<li>These actions determine what direction your character will move</li>' \
'<li>You can also choose to not input anything and not move (wait) on any turn </li>' \
'<p class="instructions_head">Tiles and Valid Moves:</p>' \
'<li>You can choose to move your character into any white or red tile </li>' \
'<li>You may also freely occupy the same tile as or pass through other players </li>' \
'<li>Black tiles are walls and you are not able to enter </li>' \
'<li>If you attempt to move into a black tile, your character will remain on the same tile </li>' \
'<p class="instructions_head">Movement Indicators:</p>' \
'<li>The last input you entered can be seen in the bottom corners of the screen </li>' \
'<li>This will be the direction your character moves when the turn timer expires </li>' \
'<li>The previous move performed by a player is shown by the wind icons behind each player</li>' \
'</ul>'


TEXT['turns'] =  '<ul>' \
'<p class="instructions_head">The Turn Timer:</p>' \
'<li>Your character will move according to the  <span style="font-weight:bold;">last action</span> you input before your <span style="font-weight:bold;">turn expires</span></li>' \
'<li>The "Turn Timer" indicates the time remaining until your move will be executed</li>' \
'<li>When the timer expires, you and your partner will then move <span style="font-weight:bold;">simultaneously</span></li>' \
'<li>Shorty after, the target will move</li>' \
'<p class="instructions_head">Remaining Moves:</p>' \
'<li>The remaining moves that your team has is shown in the top left</li>' \
'<li>You have a total of  <span style="font-weight:bold;">20 moves</span> to catch the target before the game ends</li>' \
'<li>Any remaining moves after you catch the target will be added to your final score</li>' \
'</ul>'

TEXT['survey instructions'] = '<ul>' \
'<p class="instructions_head">Please complete the survey on the following page:</p>' \
'<li>Questions relate to how you generally feel when interacting with artificial agents </li>' \
'<li>Awnser the questions to the best of your ability</li>' \
'<li>Click the bubble to record your response</li>' \
'<li>When you are finished, submit the survey to continue to the experiment instructions</li>' \
'</ul>'

TEXT['penalties'] = "<ul> <p class='instructions_head'>Risky States:</p>" + \
r"<li>During the game there will be several <span style='color: #bb0a1e;font-weight:bold;'>red tiles</span> called  <span style='color: #bb0a1e;font-weight:bold;'>'risky states'</span></li>" + \
"<li>When ending your turn in these tiles, you have a __% <span style='font-weight:bold;'>chance</span> of getting a -__ <span style='font-weight:bold;'>penalty</span> </li>" + \
"<li>You will be informed of the percent chance and penalty during the experiment  </li>" + \
"<p class='instructions_head'>Receiving a Penalty:</p>" + \
"<li>Entering a risky state does not guarantee that you will receive a penalty  </li>" + \
"<li>However, if you do receive a penalty by chance, then your screen will flash red  </li>" + \
"<li>The cumulative penalty you receive will be subtracted from your final score for that game  </li>" + \
"<li>The cumulative penalty can be seen in the top right </li>" + \
"<li>In this example, there is a 50% chance of getting a -3 penalty (will be different for you)</li>" + \
"<!--                    <li>These numbers will be different when you play</li>-->" + \
"<p class='instructions_head'>Other Player's Penalties:</p>" + \
"<li>Your partner also receives penalties in the same way you do</li>" + \
"<li>However, your and your partner's penalties are separate and do not affect each other </li>" + \
"<li>The target does not receive penalties and can move freely without consequence</li>" + \
"</ul>"

TEXT['objective'] ="<ul>" +\
"<p class='instructions_head'>Catching the Target:</p>" +\
"<li>You and your partner's objective to catch the target (<span style='color: #028a0f;font-weight:bold;'>green square</span>) </li>" +\
"<li>The target is '<span style='font-weight:bold;'>caught</span>' when you and your partner are co-located or adjacent to the target</li>" +\
"<li>You must be co-located or adjacent simultaneously for the catch to count</li>" +\
"<li>Therefore, you must cooperate with your partner because you cannot succeed alone</li>" +\
"<li>If you catch the target before you run out of moves, the game will end</li>" +\
"<p class='instructions_head'>Target Behavior & Tips for Success:</p>" +\
"<li>The target will attempt to avoid both players</li>" +\
"<li>It's goal is to move to the tile that has the highest cumulative distance to both players</li>" +\
"<li>However, the target will <span style='font-weight:bold;'>prioritize moving away</span> from the <span style='font-weight:bold;'>closest player</span> when possible </li>" +\
"<li>Therefore, it is best to trap the target in a corner and simultaneously approach it </li>" +\
"<li>This is not the only way to succeed since the target will not avoid players perfectly </li>" +\
"<li>It is ultimately up to you and your painter to decide how to best catch the target</li>" +\
"</ul>"

TEXT['practice'] = '<ul>' +\
'<p class="instructions_head">Your Participation:</p>' +\
'<li>You will play a series of 7 games each lasting a maximum of 1 minute</li>' +\
'<li>Each game will be followed by a brief survey about your interaction with your partner</li>' +\
'<li>At the end of the experiment, additional information about your participation will be provided</li>' +\
'<p class="instructions_head">Game Review:</p>' +\
'<li>Catch the target by moving next to it at the same time as your partner</li>' +\
'<li>Use the arrow keys to move your character around the screen</li>' +\
'<li>Attempt to catch the target as fast as possible to maximize your final score</li>' +\
'<li>Attempt to avoid "risky states" and penalties to maximize your final score</li>' +\
'<p class="instructions_head">Practice Round:</p>' +\
'<li>Before we begin, you will play a practice round</li>' +\
'<li>Take this time to familiarize yourself with the controls and game mechanics</li>' +\
'<li>Your performance in this practice round will not count towards your final score</li>' +\
'<li>After you finish, we will begin the experiment</li>' +\
'<li>You may go back and review previous instructions if you wish</li>' +\
'</ul>'


font_size = 50
pen_prob = 50
pen_reward = -3
TEXT['treatment'] ='<h2>Upon entering a penalty state, you have a...</h2>' + \
f'<br> <div> <label style="color: red; fontSize=50px">{pen_prob}% chance </label> <label> of getting a</label> </div>' + \
f'<br> <div> <label style="color: red;">{pen_reward} penalty </label> <label> to your final score</label> </div>'

# TEXT['treatment'] = f"You have a {pen_prob} chance of recieving a {pen_reward} penalty"

TEXT['begin instructions'] = '<ul>' \
'<li>The following pages will introduce the instructions for the game </li>' \
'</ul>' \


test_views = [
    ######## BACKGROUND ###############
    {'view':'info-frame','img': None,'title': 'Welcome!', 'content':TEXT['consent'],'fontSize':"14px",'textAlign':"left",'backButton': True},
    {'view': 'background-frame','title': 'Background', 'content': "Please provide background information"},
    {'view': 'info-frame', 'img': None, 'title': 'Survey', 'content': TEXT['survey info'],'fontSize':"14px"},
    {'view':'survey-frame'},
    ######## INSTRUCTIONS ###############
    {'view': 'info-frame', 'img': None, 'title': 'Instructions', 'content': TEXT['begin instructions'], 'backButton': False},
    {'view':'info-frame','img': GithubImgs['movement'],'title': 'Movement', 'content':TEXT['movement'] ,'backButton':True},
    {'view':'info-frame','img': GithubImgs['turns'],'title': 'Turns', 'content':TEXT['turns']},
    {'view': 'info-frame', 'img': GithubImgs['penalties'], 'title': 'Penalties', 'content': TEXT['penalties']},
    {'view': 'info-frame', 'img': GithubImgs['objective'] , 'title': 'Objective', 'content': TEXT['objective'] },
    ######## PRACTICE ###############
    {'view': 'info-frame', 'img': None, 'title': 'Practice', 'content': TEXT['practice']},
    {'view':'canvas-frame'},
    {'view': 'info-frame', 'img': None, 'title': 'Ready to Begin?', 'content': TEXT['practice'],'backButton': False},
    {'view': 'info-frame', 'img': None, 'title': 'Penalty Information', 'content': TEXT['treatment'],
     'fontSize': "50px", 'textAlign': "center", 'backButton': False},
    {'view': 'canvas-frame'},
    {'view': 'survey-frame'},
    {'view': 'info-frame', 'img': None, 'title': 'Penalty Information', 'content': TEXT['treatment'],
     'fontSize': "50px", 'textAlign': "center"},
    {'view': 'canvas-frame'},
    {'view': 'survey-frame'},
]

for igame in range(7):
    test_views.append({'view': 'info-frame', 'img': None,
                       'title': 'Penalty Information', 'content': TEXT['treatment'],
                       'fontSize': "50px", 'textAlign': "center", 'backButton': False})
    test_views.append({'view': 'canvas-frame'})
    # test_views.append({'view': 'survey-frame'})

