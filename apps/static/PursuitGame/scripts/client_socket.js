

class UserInput {
    constructor() {
        // this.no_key = {'keypress':'None'}
        // this.last_key = this.no_key
        this.input_log = {}
    }

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




