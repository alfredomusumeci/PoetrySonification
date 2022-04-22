// When the page is loaded hide the audio element.
let audioElement = document.getElementById("source");
audioElement.hidden = true;

let sonificationProduced = false;
let textInputted = "";
let notesProduced = "";

// Hide error message for unsuccessful sonification.
let vocabularyAlert = document.getElementById("vocabularyError");
vocabularyAlert.hidden = true;

// Submit form and un-hide audio element.
$(document).ready(function() {
    $("form").on("submit", function(event){
        // Prevent form from submitting.
        event.preventDefault();

        // Capture text and define post URL.
        let formValues = $(this).serialize();
        let actionUrl = '/generate_sonification';

        // Send a post request to obtain sonification of inputted text.
        $.ajax({
            type: 'POST',
            url: actionUrl,
            data: formValues,
            success: [
                // In case of success, send the generated path file to the audio element,
                // un-hide the audio element, and make a download available.
                function (data) {
                    // Temporarily store the information for optional feedback.
                    showButtons();
                    sonificationProduced = data.success;
                    textInputted = data.content;
                    notesProduced = data.notes;

                    // Set audio element.
                    audioElement.src = data.result;
                    audioElement.hidden = false;

                    // Set download button.
                    downloadButton.href = data.result;
                    downloadButton.download = "output_sonification";
                }],
            error: [
                // Otherwise, log the errors.
                function(XMLHttpRequest, textStatus, errorThrown) {
                    vocabularyAlert.hidden = false;
                    console.log(XMLHttpRequest.statusText);
                    console.log(textStatus);
                    console.log(errorThrown);
            }],
        });
    });
});