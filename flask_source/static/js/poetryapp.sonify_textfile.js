// Grab upload button.
const uploadElement = document.getElementById("formFile");
uploadElement.addEventListener("change", handleFiles);

// Hide error message for wrong type at start.
let wrongTypeAlert = document.getElementById("wrongTypeError");
wrongTypeAlert.hidden = true;

/**
 * Reads the uploaded file and calls the sonify function.
 * @param input - The file input element.
 */
function handleFiles(input) {
    let file = input.target.files[0];

    if (isText(file.name)) {
        let reader = new FileReader();

        reader.readAsText(file);

        reader.onload = function () {
            textArea.value = reader.result;
            textArea.click();
        };

        reader.onerror = function () {
            console.log(reader.error);
        };
    } else {
        // If wrong format show error message.
        wrongTypeAlert.hidden = false;
    }
}

/**
 * Returns the filetype extension of a file.
 * @param filename - The filename.
 * @returns {string} - The filetype extension.
 */
function getExtension(filename) {
    let filenameSplit = filename.split(".");
    return filenameSplit[filenameSplit.length - 1];
}

/**
 * Returns true if the filetype is text.
 * @param filename - The filename.
 * @returns {boolean} - True if the filetype is text.
 */
function isText(filename) {
    let extension = getExtension(filename);
    switch (extension.toLowerCase()) {
        case "txt" : return true;
        // more can be added...
    }
    return false;
}