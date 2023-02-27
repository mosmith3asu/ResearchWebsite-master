




// ###############################################
// ########### INTERFACE #########################
// ###############################################
// let content_frames = document.getElementById("contentViewPort").children;

class UserInput {
    constructor() { this.input_log = {}}
    store (key,val){
        // console.log(key,val)
        this.input_log[key] = val;
    }
    read_buffer(){
        // const res = this.last_key
        const res = this.input_log
        // console.log(res)
        this.input_log = {}
        // delete thisIsObject["Cow"]
        // this.last_key = this.no_key
        return res
    }
}

function decode_keypress(e) {
    if (e.keyCode === 37) { return pressed = 'left'
    } else if (e.keyCode === 38) {return 'up' //up
    } else if (e.keyCode === 39) {return 'right' //right
    } else if (e.keyCode === 40) {return 'down' //down
    } else if (e.keyCode === 32) {return 'spacebar'
    }  else {   return 'other'}

}

// Disable scroll
document.addEventListener("keydown", function(event) {
    if (event.code === "Space" || event.code === "ArrowUp" || event.code === "ArrowDown") { event.preventDefault(); }
});

// Read Key Inputs
$(document).keydown(function(e) {
    console.log(`PRESS: ${decode_keypress(e)}`);
    user_input.store('keypress',decode_keypress(e));
})



function hide_all(){
    for (const frame of content_frames) {frame.style.display = 'none';}
}
function show(sel) {
    console.log(`showing:${sel}`)
    hide_all();
    document.getElementById(sel).style.display = 'inline-grid';
}
show(current_view)
let user_input = new UserInput()

let G = new Game([1,0,3,3,1,0]);
G.render()


// ###############################################
// ########### WEBSOCKET #########################
// ###############################################


$(document).ready(function() {
    const socket = io();



    // Emit actions on .js event ######################################################
    setInterval(function (){
        const user_input_buffer =  user_input.read_buffer()
        socket.emit('update_gamestate', user_input_buffer)
        console.log(`Client Send: ${user_input_buffer}`)

    }, update_rate) // 1 millisecond is almost close to continue


    // Update game data on response from server ##########################################
    socket.on( 'update_gamestate', (data)=>{
        // console.log(data)
        // NEW VIEW => Hide all views for init


        if (data['view'] !== current_view){
            show(data['view'])
            current_view = data['view']
            console.log('Changing view to '+ current_view)
        }

        if (data['view'] === 'canvas-frame'){
            G.update(data)
            G.render()
            document.getElementById("backButton").style.display = 'none';
        }

        // if (data['view'] === 'info-frame'){
        //     infopage.update(data)
        //
        //     console.log('udating info-frame')
        // } else if (data['view'] === 'background-frame'){
        //     backgroundpage.update(data)
        // } else if (data['view'] === 'canvas-frame'){
        //     G.update(data)
        //     G.render()
        // }

    })
})
