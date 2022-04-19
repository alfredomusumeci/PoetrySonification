$(document).ready(function() {
    // When form is submitted ...
    submitButton.addEventListener("click", function() {
        // Disable button and add spinner.
        $(this).disabled = true;
        $(this).html(
        '<span id="spinner" class="spinner-border" role="status" aria-hidden="true"></span>'
        );
    });

    // When song is loaded ...
    audioElement.addEventListener("load", function() {
        // Enable button again, remove spinner and restore inner text.
        submitButton.disabled = false;
        $('#spinner').remove()
        submitButton.innerText = "Sonify"
    });
});