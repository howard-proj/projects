// Game values
let min = 1,
    max = 10,
    winningNum = getRandomNum(min, max),
    guessesLeft = 3;

// UI Elements
const UI_game = document.querySelector('#game'),
    UI_minNum = document.querySelector('.min-num'),
    UI_maxNum = document.querySelector('.max-num'),
    UI_guessBtn = document.querySelector('#guess-btn'),
    UI_guessInput = document.querySelector('#guess-input'),
    UI_message = document.querySelector('.message');

// Assign UI min and max; it will show in the html
UI_minNum.textContent = min;
UI_maxNum.textContent = max;

// Play again event listener
game.addEventListener('mousedown', function(e){
    if(e.target.className === 'play-again') {
        window.location.reload();
    }
});

// Listen for guess
UI_guessBtn.addEventListener('click', function(){
    let guess = parseInt(UI_guessInput.value);

    // Validate
    if(isNaN(guess) || guess < min || guess > max) {
        setMessage(`Please enter a number between ${min} and ${max}`, 'red');
    }

    // Check if won
    if(guess === winningNum){
        // Game over - won
        gameOver(true, `${winningNum} is correct! YOU WIN!`);

    } else {
        // Wrong Number
        guessesLeft -= 1;

        if (guessesLeft === 0) {
            // Game over - lost
            gameOver(false, `Game Over, you lost. The correct number was ${winningNum}`);
        }
        else {
            // Game continues - answer wrong

            // Change color
            UI_guessInput.style.borderColor = 'red';

            // Tell user its not the correct number
            setMessage(`${guess} is not correct, ${guessesLeft} guesses left`, 'red');

            // Clear input
            UI_guessInput.value = '';
        }
    }
});

// Game over
function gameOver(won, msg) {
    let color;
    won === true? color='green' : color='red';

    // Disable input
    UI_guessInput.disabled = true;
    // Change color
    UI_guessInput.style.borderColor = color;
    // Set text color
    UI_message.style.color = color;
    // Set message
    setMessage(msg, color);

    // Play Again?
    UI_guessBtn.value = 'Play Again';
    UI_guessBtn.className += 'play-again';
}

// Get Winning Number
function getRandomNum(min, max) {
    return (Math.floor(Math.random() * (max - min + 1) + min));
}

// Set message
function setMessage(msg, color) {
    UI_message.style.color = color;
    UI_message.textContent = msg;
}