// Defining class using es6
//     class Vehicle {
//       constructor(name, maker, engine) {
//         this.name = name;
//         this.maker =  maker;
//         this.engine = engine;
//       }
//       getDetails(){
//           return (`The name of the bike is ${this.name}.`)
//       }
//     }
//     // Making object with the help of the constructor
//     let bike1 = new Vehicle('Hayabusa', 'Suzuki', '1340cc');
//     let bike2 = new Vehicle('Ninja', 'Kawasaki', '998cc');
//
//     console.log(bike1.name);    // Hayabusa
//     console.log(bike2.maker);   // Kawasaki
//     console.log(bike1.getDetails());
//
//
class Game  {
    constructor(state) {
        this.world = 0;
        this.state = [1,1,1,1,1,1];
        this.timer = -0.5;
        this.pen_alpha = 0.0
        this.nPen = 0
        this.moves = 20;
        this.playing = true;
        this.is_finished = false;
        this.penalty_states = [[1,1]]


        this.can = document.getElementById("canvas");
        this.ctx = this.can.getContext("2d");
        this.nCol = 7; this.w = this.can.width;
        this.nRow = 7; this.h = this.can.height;

        this.world_data = [
            [1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1],
            [1,0,1,0,1,0,1],
            [1,0,0,0,0,0,1],
            [1,0,1,0,1,0,1],
            [1,0,0,0,0,0,1],
            [1,1,1,1,1,1,1]];


        this.c_player = 'rgba(0,0,255, 1.0)'
        this.c_partner = 'rgba(100,100,255, 1.0)'
        this.c_evader = 'rgba(0,255,0, 1.0)'

    }
    render(){
        this.clear()
        this.draw_world()
        this.draw_players()
        this.draw_move_counter()
        this.draw_penalty_counter()
        this.draw_timers()

        this.draw_penalty_overlay()
        this.draw_finished_overlay()
    }
    clear(){ this.ctx.clearRect(0,0,this.w,this.h);}
    draw_world() {
        var c_black = 'rgba(0,0,0, 1.0)'
        var c_red = 'rgba(255,0,0, 0.3)'
        var c_white ='rgba(255,255,255, 1.0)'
        var scale= 1.01
        var nCol = 7; var nRow = 7;  var col = 0;
        var w = this.w/this.nCol;
        var h = this.h/this.nRow;
        var i = 0;  var j = 0;
        var r; var c;
        var is_num;

        // Update world data with penalties
        var e = 0;
        for (i = 0; i < this.penalty_states.length; i++) {
            r = this.penalty_states[i][0]
            c = this.penalty_states[i][1]
            this.world_data[r][c] = 2
        }


        // DRAW ARRAY ###########################
        for (i = 0; i < nRow; i++) {
            for (j = 0, col = nCol; j <= col; j++) {
                var val = this.world_data[i][j]
                if (val===0) {
                    console.log((`Drawing ${j}${i} ${val} EMPTY`))
                    this.ctx.fillStyle = c_white //empty
                    this.ctx.fillRect(j * w, i * h, scale * w, scale * h)
                }else if (val===1){
                     console.log((`Drawing ${j}${i} ${val} PENALTY`))
                    this.ctx.fillStyle = c_black //penalty
                    this.ctx.fillRect(j * w, i * h, scale*w, scale*h)
                }else if (val===2){
                     console.log((`Drawing ${j}${i} ${val} PENALTY`))
                    this.ctx.fillStyle = c_red //penalty
                    this.ctx.fillRect(j * w, i * h, scale*w, scale*h)
                }
            }
        }
        return (`World
              ${this.state} ${this.move}`)
    }
    draw_players() {

        var sz_player = 0.8
        var sz_partner = 0.6
        var sz_evader = 0.9
        var r; var c;
        var w = this.w/this.nCol;
        var h = this.h/this.nRow;
        var loc = this.state

          // EVADER  #############################\
        r = loc[4]*h + (h-sz_evader*h)/2 // row
        c = loc[5]*w + (w-sz_evader*w)/2 // col
        this.ctx.fillStyle = this.c_evader //empty
        this.ctx.fillRect(c,r, sz_evader*w, sz_evader*h)

        // PLAYER #############################\
        r = loc[0]*h + (h-sz_player*h)/2 // row
        c = loc[1]*w + (w-sz_player*w)/2 // col
        this.ctx.fillStyle = this.c_player //empty
        this.ctx.fillRect(c,r,sz_player*w, sz_player*h)

        // Partner #############################\
        r = loc[2]*h + (h-sz_partner*h)/2 // row
        c = loc[3]*w + (w-sz_partner*w)/2 // col
        this.ctx.fillStyle = this.c_partner //empty
        this.ctx.fillRect(c,r, sz_partner*w, sz_partner*h)

    }
    draw_finished_overlay(){
        if (this.is_finished){
            var c_overlay = 'rgba(0,0,0,0.2)';
            var c_text = 'rgba(255,255,255,1.0)';
            this.ctx.fillStyle = c_overlay;
            // this.ctx.fillRect(0, 0, can.width, can.height);
            this.ctx.fillRect(0, 0, this.w, this.h);

            this.ctx.font = '48px serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillStyle = c_text;
            this.ctx.fillText('Game',    350, 300);
            this.ctx.fillText('Complete',350, 350);
        }
    }
    draw_penalty_overlay(){
        // if (this.in_pen){
        var c_overlay = `rgba(255,0,0,${this.pen_alpha})`;
        this.ctx.fillStyle = c_overlay;
        this.ctx.fillRect(0, 0, this.w, this.h);
        // }
    }
    draw_penalty_counter(){
        var yloc = 662
        var c_text = 'rgba(200,0,0,1.0)';
        this.ctx.font = '30px serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = c_text;
        this.ctx.fillText('Penalty: ',  300, yloc);
        this.ctx.fillText(this.nPen, 375, yloc);
    }
    draw_move_counter(){
        var yloc = 62
        var xlabel = 300
        var xcounter = 375
        var h_patch = 90

        var c_text = 'rgba(255,255,255,1.0)';
        var c_bg = 'rgba(0,0,0,1.0)';
//    c_bg = 'rgba(255,255,0,1.0)';

        // Clear counter area
        this.ctx.fillStyle = c_bg;
        this.ctx.clearRect(0,0,this.w,h_patch)
        this.ctx.fillRect(0,0,this.w,h_patch)

        // Add counter elements in
        this.ctx.font = '30px serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = c_text;
        this.ctx.fillText('Moves: ',  xlabel, yloc);
        this.ctx.fillText(this.moves, xcounter, yloc);

        //ctx.fillRect(0, 0, can.width, can.height);

    }
    draw_timers(){
        var timer_val = this.timer
        var y_start = 100
        var y_end = 600
        var timer_height = y_end-y_start
        var timer_width = 60
        var x_evader = 50
        var x_player = 650
        var c_fill =  'rgba(0,0,0, 1.0)'
        var c_player_dim = 'rgba(0,0,150, 1.0)'
        var c_evader_dim = 'rgba(0,150,0, 1.0)'

        // Clear timer space
        this.ctx.clearRect(x_evader-timer_width/2, y_start, timer_width, timer_height)
        this.ctx.clearRect(x_player-timer_width/2, y_start, timer_width, timer_height)
        this.ctx.fillStyle = c_fill //empty
        this.ctx.fillRect(x_evader-timer_width/2, y_start, timer_width, timer_height)
        this.ctx.fillRect(x_player-timer_width/2, y_start, timer_width, timer_height)

        if (timer_val<=0){
            var prog = (1+timer_val)*timer_height
            var tHeight =y_start+prog

            // Evader Timer --------------------------
            this.ctx.fillStyle = this.c_evader //empty
            this.ctx.fillRect(x_evader-timer_width/2, tHeight, timer_width, y_end-tHeight)

            // Disabled player timer -----------------
            this.ctx.fillStyle = c_player_dim
            this.ctx.fillRect(x_player-timer_width/2, y_start, timer_width, timer_height)

        }else{
            prog = timer_val*timer_height
            tHeight = y_start+prog

            // player timer --------------------------
            this.ctx.fillStyle = this.c_player
            this.ctx.fillRect(x_player-timer_width/2, tHeight, timer_width, y_end-tHeight)

            // Disabled Evader Timer -----------------
            this.ctx.fillStyle = c_evader_dim //empty
            this.ctx.fillRect(x_evader-timer_width/2, y_start, timer_width,timer_height)
        }
    }

} // End of Class Game


let G = new Game([1,0,3,3,1,0]);
G.render()

$(document).ready(function() {
    const socket = io();
    let G = new Game(1,[2,1,3,3,5,5]);

    // Emit actions on .js event ######################################################
    $('.left').click(function(){ if (G.playing) socket.emit('next_pos', {'action':'left'}) })
    $('.right').click(function(){ if (G.playing) socket.emit('next_pos', {'action':'right'}) })
    $('.up').click(function(){  if (G.playing) socket.emit('next_pos', {'action':'up'}) })
    $('.down').click(function(){  if (G.playing) socket.emit('next_pos', {'action':'down'}) })

    // Update game data on response from server ##########################################
    socket.on('update_gamestate', (data)=>{
        G.world = data['iworld'];
        G.state =  data['state'];
        G.timer = data['timer'];
        G.pen_alpha = data['pen_alpha'];
        G.nPen = data['nPen'];
        G.moves = data['moves'];
        G.playing = data['playing'];
        G.is_finished = data['is_finished'];
        G.penalty_states = data['penalty_states'];
        G.render()
    })
    /* ################ DETECTING ARROW KEYPRESS ################ */

     $(document).keydown(function(e) {
            if (e.keyCode === 37) {  if (playing) socket.emit('next_pos', {'action':'left'})
            } else if (e.keyCode === 38) {  if (playing) socket.emit('next_pos', {'action':'up'}) //up
            } else if (e.keyCode === 39) { if (playing) socket.emit('next_pos', {'action':'right'}) //right
            } else if (e.keyCode === 40) {  if (playing) socket.emit('next_pos', {'action':'down'}) //down
            }
        })
    })
//
//
//
//     // console.log(G.preview_fun());
//      console.log(G.drawWorld());
//  console.log(G.draw_players());
//   console.log(G.draw_finished_overlay());
//   G.draw_penalty_overlay()
//  G.draw_penalty_counter()
// G.draw_move_counter()
// G.draw_timers()
// // console.log(Game.phone_number.landline);