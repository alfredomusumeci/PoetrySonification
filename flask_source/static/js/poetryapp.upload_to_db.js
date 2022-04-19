let acceptButton = document.getElementById("accept-button");
let denyButton = document.getElementById("deny-button");
let modalBody = document.getElementById("feedbackModalBody");

acceptButton.addEventListener("click", function (event) {
    return uploadToDatabase(event.target.id)})
denyButton.addEventListener("click", function (event) {
    return uploadToDatabase(event.target.id)})

/**
 * Upload the sonification (lyrics, notes) to the database.
 * @param id - The id of the button that was clicked.
 */
function uploadToDatabase(id) {
    renderConfirmation(id)

    // Upload only if the user likes the sonification.
    if (id === "accept-button") {
        if (sonificationProduced) {
            console.log(textInputted)
            console.log(notesProduced)

            $.ajax({
              type: "POST",
              url: "/upload_to_db",
              data: {"text": textInputted, "notes": notesProduced}
            }).done(function(data) {
               console.log(data)
            });
        }
    }
}

/**
 * Render the confirmation (or cancellation) image.
 * @param id - The id of the button that was clicked.
 */
function renderConfirmation(id) {
    if (id === "deny-button") {
        const src = "static/svg/cancel.svg";
        const alt = "Cancellation received"
        const p = "Maybe next time!";
        injectImgHTML(src, alt, p);
    } else if (id === "accept-button") {
        const src = "static/svg/confirm.svg";
        const alt = "Confirmation received"
        const p = "Thank you!";
        injectImgHTML(src, alt, p)
    }
}

/**
 * Inject the confirmation image into the modal body.
 * @param source - The source of the image.
 * @param description - The description of the image.
 * @param text - The text to be displayed.
 */
function injectImgHTML(source, description, text) {
    // Create the image element.
    const img = document.createElement("img");
    img.className = "centered";
    img.src = source;
    img.alt = description;
    img.style.width = "100px";
    img.style.height = "100px";

    // Disable the buttons.
    acceptButton.disabled = true;
    denyButton.disabled = true;

    // Insert the image into the modal body along with the text.
    modalBody.insertBefore(img, modalBody.children[0]);
    const p = modalBody.children[1];
    p.innerText = text;
}

// Recreate the modal body every time the modal is opened.
$(document).ready(function() {
  $("#feedbackModal").on("hidden.bs.modal", function() {
    $("#feedbackModalBody").html("<p align=\"justify\" style='margin-bottom: 0'> " +
        "Do you think the music produced reflected the feelings behind the inputted text?\n" +
        "Your answer will be used for further improvement of the model.\n" +
        "<br><b> Positive answers will have their text and corresponding music recorded. </b>\n" +
        "</p>");

    acceptButton.disabled = false;
    denyButton.disabled = false;
  });
});