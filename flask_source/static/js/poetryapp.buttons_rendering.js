// Grab the elements needed
let textArea = document.getElementById("poetry-content");
let submitButton = document.getElementById("submit-button");
let showSheetButton = document.getElementById("music-sheet-button");
let rateButton = document.getElementById("rate-button");
let downloadButton = document.getElementById("download-button");

let invalidInput = true;

// Set the initial button state to disabled (i.e., upon loading)
showSheetButton.hidden = invalidInput;
rateButton.hidden = invalidInput;
downloadButton.hidden = invalidInput;
submitButton.disabled = invalidInput;

// Check if input is valid and update variable accordingly
$("#poetry-content").on("input click", validateInput);

/**
 * Validates the input of the text area.
 * @returns {boolean} True if the input is valid, false otherwise.
 */
function validateInput() {
    invalidInput = textArea.value.trim().replace(/\s/g, "") === "";
    submitButton.disabled = invalidInput;
}

/**
 * Show the buttons if the input is valid and submit button has been pressed.
 */
function showButtons() {
    showSheetButton.hidden = invalidInput;
    rateButton.hidden = invalidInput;
    downloadButton.hidden = invalidInput;
}