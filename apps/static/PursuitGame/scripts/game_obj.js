class ColorPallet {
    constructor() {
        this.white =  'rgba(255,255,255,1.0)'
        this.black =  'rgba(0,0,0,1.0)'
        this.red =  'rgba(200,0,0,1.0)';
        this.light_red =  'rgba(200,100,100,1.0)';
    }
}
let COLORS = new ColorPallet()



//###################################################
//###################################################
//###################################################
function center_align_element(element){
    element.style.display = "block";
    element.style.margin = 'auto';
    element.style.height = '100%';
}


class Player  {
    constructor(pos,color,size){
        // this.pos = [0,0]
        this.last_pos = pos
        this.last_rot = 90
        this.color = color
        this.can = document.getElementById("canvas");
        this.ctx = this.can.getContext("2d");
        this.nCol = 7; this.nRow = 7;
        this.tile_w = this.can.width/this.nCol;
        this.tile_h = this.can.height/this.nRow;
        this.def_scale = size
        this.scale = size
        this.fill = true
        this.unfilled_lw = 8
    }
    draw (pos) {

        var c_ypx = pos[0]*this.tile_h +this.tile_h/2;  // center y loc in pixels
        var c_xpx = pos[1]*this.tile_w +this.tile_w/2; // center x loc in pixels
        var p_ypx = this.scale*this.tile_h;
        var p_xpx =  this.scale*this.tile_w;

        var degree = this.pos2rot(pos)
        var deg2rad = Math.PI / 180


        this.ctx.translate(c_xpx,c_ypx)
        this.ctx.rotate(degree * deg2rad)
        this.ctx.beginPath();
        this.ctx.moveTo(-p_xpx/2 ,-p_ypx/2); // top left
        this.ctx.lineTo(0,         p_ypx/2); // bottom tip
        this.ctx.lineTo(p_xpx/2 , -p_ypx/2 ); // top right
        this.ctx.lineTo(0,        -p_ypx/4);// center indent
        this.ctx.closePath();
        if (this.fill){
            this.ctx.fillStyle = this.color //empty
            this.ctx.fill();
        } else {
            this.ctx.lineWidth = this.scale*this.unfilled_lw
            this.ctx.strokeStyle = this.color
            this.ctx.stroke();
        }
        this.ctx.rotate(-degree * deg2rad)
        this.ctx.translate(-c_xpx,-c_ypx)
        this.last_pos = pos;

    }
    pos2rot(pos){
        var deg;
        var dx = pos[1]-this.last_pos[1] ;
        var dy = pos[0]-this.last_pos[0] ;
        if (dx === 0 && dy ===0){ deg = this.last_rot }
        else if (dy === 1  && dx === 0  ){  deg = 0} // up
        else if (dy === -1 && dx === 0  ){ deg = 180  } // down
        else if (dy === 0  && dx === 1  ){  deg = 270 } // right
        else if (dy === 0  && dx === -1 ){ deg = 90  } // left
        else{ deg = this.last_rot
            console.error((`Unkown Player Move:${deg} - ${pos[0]} ${pos[1]}  => ${this.last_pos[0]} ${this.last_pos[1]}  ${dy} ${dx} `))
        }
        this.last_rot = deg
        return deg

    }

}
class Game  {
    constructor(state) {
        this.world = 0;
        this.state = [1,1,1,1,1,1];
        this.timer = 1;
        this.pen_alpha = 0.0
        this.nPen = 0
        this.moves = 20;
        this.playing = true;
        this.is_finished = false;
        this.is_closed = false;
        this.penalty_states = [[1,1]]
        this.current_action = 4

        const canvasFrame = document.getElementById("canvas-frame");
        center_align_element(canvasFrame)
        // canvasFrame.style.width = '100%'
        // canvasFrame.style.height = '100%'
        canvasFrame.style.width = '700px'
        canvasFrame.style.height = '700px'
        canvasFrame.style.backgroundColor = 'black'
        canvasFrame.style.position = 'absolute'
        canvasFrame.style.border = '1px solid blue'



        const canvas = document.getElementById("canvas");
        center_align_element(canvas)
        canvas.style.backgroundColor = 'white'
        canvas.style.border = '1px solid blue'


        this.can = canvas
        this.ctx = this.can.getContext("2d");
        this.nCol = 7; this.can_w = this.can.width; this.tile_w = this.can.width/this.nCol;
        this.nRow = 7; this.can_h = this.can.height; this.tile_h = this.can.height/this.nRow;

        this.empty_world = [
            [1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1],
            [1,0,1,0,1,0,1],
            [1,0,0,0,0,0,1],
            [1,0,1,0,1,0,1],
            [1,0,0,0,0,0,1],
            [1,1,1,1,1,1,1]];
        this.world_data = [
            [1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1],
            [1,0,1,0,1,0,1],
            [1,0,0,0,0,0,1],
            [1,0,1,0,1,0,1],
            [1,0,0,0,0,0,1],
            [1,1,1,1,1,1,1]];
        this.load_empty_world()

        this.c_robot = 'rgba(100,100,255, 1.0)'
        this.c_human = 'rgba(0,0,255, 1.0)'
        this.c_evader = 'rgba(0,255,0, 1.0)'

        this.robot = new Player([0,0], this.c_robot,0.6)
        this.robot.fill = false
        this.human = new Player([0,0], this.c_human,0.8)
        this.evader = new Player([0,0], this.c_evader,0.8)

    }
    render(){
        if (! this.is_closed){

            this.clear()
            this.draw_world()
            this.draw_players()

            this.draw_penalty_counter()
            this.draw_timers()

            this.draw_move_counter()
            this.draw_iworld_header()
            this.draw_current_action()
            this.draw_penalty_overlay()
            this.draw_finished_overlay()
            if(this.is_finished){
                console.log('Posting close')
                this.post_close()
            }
        }


    }
    update(data) {
        // console.log(data['current_action'])
        this.world = data['iworld'];
        this.state =  data['state'];
        this.timer = data['timer'];
        this.pen_alpha = data['pen_alpha'];
        this.nPen = data['nPen'];
        this.moves = data['moves'];
        this.playing = data['playing'];
        this.is_finished = data['is_finished'];
        const pen_states =  data['penalty_states'];
        this.penalty_states = pen_states;
        this.current_action = data['current_action'];
    }
    clear(){
        this.load_empty_world()
        this.ctx.clearRect(0,0,this.can_w,this.can_h);}
    draw_world() {
        var c_black = 'rgba(0,0,0, 1.0)'
        var c_red = 'rgba(255,0,0, 0.3)'
        var c_white ='rgba(255,255,255, 1.0)'
        var scale= 1.01
        var nCol = 7; var nRow = 7;  var col = 0;
        var w = this.tile_w;
        var h = this.tile_h;
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
                    // console.log((`Drawing ${j}${i} ${val} EMPTY`))
                    this.ctx.fillStyle = c_white //empty
                    this.ctx.fillRect(j * w, i * h, scale * w, scale * h)
                }else if (val===1){
                    // console.log((`Drawing ${j}${i} ${val} PENALTY`))
                    this.ctx.fillStyle = c_black //penalty
                    this.ctx.fillRect(j * w, i * h, scale*w, scale*h)
                }else if (val===2){
                    // console.log((`Drawing ${j}${i} ${val} PENALTY`))
                    this.ctx.fillStyle = c_red //penalty
                    this.ctx.fillRect(j * w, i * h, scale*w, scale*h)
                }
            }
        }
        return (`World
              ${this.state} ${this.move}`)
    }
    draw_players() {
        var loc = this.state
        this.evader.fill = !(loc[2] === loc[4] && loc[3] === loc[5]);


        this.human.draw([loc[2],loc[3]])
        this.evader.draw([loc[4],loc[5]])
        this.robot.draw([loc[0],loc[1]])
    }
    draw_finished_overlay(){
        if (! this.playing){
            var c_overlay = 'rgba(0,0,0,0.2)';
            var c_text = 'rgba(255,255,255,1.0)';
            this.ctx.fillStyle = c_overlay;
            // this.ctx.fillRect(0, 0, can.width, can.height);
            this.ctx.fillRect(0, 0, this.can_w, this.can_h);

            this.ctx.font = '48px serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillStyle = c_text;
            this.ctx.fillText('Game',    350, 300);
            this.ctx.fillText('Complete',350, 350);
        }
    }
    draw_penalty_overlay(){
        this.ctx.fillStyle = `rgba(255,0,0,${this.pen_alpha})`;
        this.ctx.fillRect(0, 0, this.can_w, this.can_h);
    }
    draw_penalty_counter(){
        var font_h = 30
        // var yloc = (this.tile_h * (this.nRow-1) )+ 0.7*this.tile_h + font_h/2//(this.nRow/2 -0.4)
        var yloc = (this.tile_h * (this.nRow-1) )+ this.tile_h/4 + font_h/2//(this.nRow/2 -0.4)
        var xloc = this.tile_w * (5)
        var xoff_cnter =  0.75*this.tile_w

        var hpad = 0.05*this.tile_h
        var h_patch = 1.0*font_h//0.5*this.tile_h
        var w_patch =0.5*this.tile_w


        var c_label = COLORS.red
        var c_counter = COLORS.black
        var c_patch = COLORS.light_red

        // Clear counter area
        this.ctx.fillStyle = c_patch;
        this.ctx.clearRect((xloc+xoff_cnter)-w_patch/2,yloc-h_patch + hpad,w_patch,h_patch)
        this.ctx.fillRect((xloc+xoff_cnter)-w_patch/2,yloc-h_patch + hpad,w_patch,h_patch)

        // Add counter elements in
        this.ctx.font = (`${font_h}px serif`)// '30px serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = c_label;
        this.ctx.fillText('Penalty: ',  xloc, yloc);

        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = c_counter;
        this.ctx.fillText(this.nPen, xloc+xoff_cnter, yloc);
    }
    draw_current_action(){
        var headlen = this.tile_h*0.25; // length of head in pixels
        var arrow_color = COLORS.white
        var fromy = (this.nRow-0.5)*this.tile_h
        var fromx = (this.nCol/2)*this.tile_w
        var arrow_len = 0.3*this.tile_h
        let DIRECTION = [[0,arrow_len],[-arrow_len,0],[0,-arrow_len],[arrow_len,0],[0,0]]
        var dy = DIRECTION[this.current_action][1]
        var dx = DIRECTION[this.current_action][0]

        if (dx === 0 && dy ===0){
            var font_h = 30
            this.ctx.font = (`${font_h}px serif`)// '30px serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillStyle = arrow_color;
            this.ctx.fillText('[...]', fromx, fromy+font_h/2);
        } else {
            var tox = fromx + dx;
            var toy = fromy + dy;
            fromx = fromx - dx;
            fromy = fromy - dy;

            var angle = Math.atan2(dy, dx);
            this.ctx.beginPath();
            this.ctx.moveTo(fromx, fromy);
            this.ctx.lineTo(tox, toy);
            this.ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
            this.ctx.moveTo(tox, toy);
            this.ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
            this.ctx.strokeStyle = arrow_color
            this.ctx.stroke();
        }

    }
    draw_iworld_header(){

        var font_h = 50
        var yloc = (this.tile_h * 0 )+ 0.4*this.tile_h + font_h/2
        var xloc = 3.5*this.tile_w
        var xoff_cnter =  0.75*this.tile_w

        var hpad = 0.05*this.tile_h
        var h_patch = 1.0*font_h//0.5*this.tile_h
        var w_patch = this.can_w


        var c_label = COLORS.white//'rgba(255,255,255,1.0)'
        var c_patch = COLORS.black//'rgba(0,0,0,1.0)'

        // Clear counter area
        this.ctx.fillStyle = c_patch;
        this.ctx.clearRect((xloc+xoff_cnter)-w_patch/2,yloc-h_patch + hpad,w_patch,h_patch)
        this.ctx.fillRect((xloc+xoff_cnter)-w_patch/2,yloc-h_patch + hpad,w_patch,h_patch)

        // Add counter elements in
        this.ctx.font = (`${font_h}px serif`)// '30px serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = c_label;
        this.ctx.fillText(`GAME ${this.world}/7`,  xloc, yloc);


    }
    draw_move_counter(){
        var font_h = 30
        var yloc = (this.tile_h * (this.nRow-1) )+ this.tile_h/4 + font_h/2//(this.nRow/2 -0.4)
        var xloc = this.tile_w * (1)
        var xoff_cnter =  0.75*this.tile_w

        var hpad = 0.05*this.tile_h
        var h_patch = 1.0*font_h//0.5*this.tile_h
        var w_patch =0.5*this.tile_w


        var c_label = 'rgba(255,255,255,1.0)'
        var c_counter = 'rgba(0,0,0,1.0)'
        var c_patch = 'rgba(255,255,255,1.0)'

        // Clear counter area
        this.ctx.fillStyle = c_patch;
        this.ctx.clearRect((xloc+xoff_cnter)-w_patch/2,yloc-h_patch + hpad,w_patch,h_patch)
        this.ctx.fillRect((xloc+xoff_cnter)-w_patch/2,yloc-h_patch + hpad,w_patch,h_patch)

        // Add counter elements in
        this.ctx.font = (`${font_h}px serif`)// '30px serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = c_label;
        this.ctx.fillText('Moves: ',  xloc, yloc);

        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = c_counter;
        this.ctx.fillText(this.moves, xloc+xoff_cnter, yloc);

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
            this.ctx.fillStyle = this.c_human
            this.ctx.fillRect(x_player-timer_width/2, tHeight, timer_width, y_end-tHeight)

            // Disabled Evader Timer -----------------
            // this.ctx.fillStyle = c_evader_dim //empty
            // this.ctx.fillRect(x_evader-timer_width/2, y_start, timer_width,timer_height)
            this.ctx.fillStyle = this.c_human //empty
            this.ctx.fillRect(x_evader-timer_width/2, tHeight, timer_width, y_end-tHeight)

        }
    }

    load_empty_world(){
        for(let r=0; r<7;r++){
            for(let c=0; c<7;c++){
                this.world_data[r][c] = this.empty_world[r][c]
            }
        }
    }
    post_close(){
        // $.post(render_route, {advance: 1 });
        user_input.store('button','continue')
        this.ctx.clearRect(0,0,this.can_w,this.can_h);
        if (game_end_redirect){location.replace(render_route)}
        // this.is_closed = true;
        this.load_empty_world()
    }

} // End of Class Game

