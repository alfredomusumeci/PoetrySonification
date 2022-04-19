let audioElement = document.getElementById("source");

window.onload = function () {
    let canvas = document.getElementById("audio_visual");

    // The contexts for the canvas element.
    let context = canvas.getContext("2d");
    let audioContext = new AudioContext();
    let source = audioContext.createMediaElementSource(audioElement);
    audioContext.state = "suspended";

    // The analyser used to get the frequency data.
    let analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;

    // Connecting all the elements.
    source.connect(analyser);
    source.connect(audioContext.destination);

    // The frequency data.
    let dataArray = new Uint8Array(analyser.frequencyBinCount);

    /**
     * The function that is called every time the frequency data is updated.
     */
    function renderFrame() {
        requestAnimationFrame(renderFrame);
        analyser.getByteFrequencyData(dataArray);
        draw(dataArray);
    }

    /**
     * The function that draws the frequency data on the canvas.
     * @param data - The frequency data.
     */
    function draw(data) {
        // Unsigned int is converted to array.
        data = [...data];

        // Clear the last drawing.
        context.clearRect(0, 0, canvas.width, canvas.height);

        // Draw again.
        let space = canvas.width / data.length;
        data.forEach((value, i) => {
            context.beginPath();
            context.moveTo(space * i, canvas.height);
            context.lineTo(space * i, canvas.height - value);
            context.stroke();
        })
    }

    renderFrame();

    // Start or stop the drawing.
    audioElement.onclick = () => {
        if(audioContext.state === 'running') {
            audioContext.suspend().then(function() {
            audioElement.textContent = 'Resume context';
            });
        } else if(audioContext.state === 'suspended') {
            audioContext.resume().then(function() {
            audioElement.textContent = 'Suspend context';
            });
        }
    }
}