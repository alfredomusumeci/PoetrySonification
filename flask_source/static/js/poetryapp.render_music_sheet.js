// Create a new Music Sheet visualizer.
let osmdElement = new opensheetmusicdisplay.OpenSheetMusicDisplay("osmdContainer")

// Set visualization options.
osmdElement.setOptions({
    backend: 'svg',
    drawTitle: false,
    drawingParameters: 'compacttight',
});

// Generate the music sheet from a musix_xml file upon
// loading of the respective modal window.
$('#musicSheetModal').on('shown.bs.modal', composeSheet);

/**
 * Generate the music sheet from a musix_xml file.
 */
function composeSheet() {
    osmdElement
        .load('../static/assets/music-sheet-xml/output' + document.cookie.split('=')[1] + '.musicxml')
        .then(() => osmdElement.render())
        .then(() => osmdElement.exportSVG());

    // When the modal is pressed, a link is generated at the bottom
    // of the page. Grab and delete that to use the download button instead.
    // Use an observer to do this as the object gets dynamically generated.
    const observer = new MutationObserver((mutations, obs) => {
      const downloadLink = document.querySelector('body > a');
      if (downloadLink) {
          let href = downloadLink.href;
          let download = downloadLink.download
          downloadLink.remove();
          document.querySelector('#downloadButton').href = href
          document.querySelector('#downloadButton').download = download

        obs.disconnect();
      }
    });

    observer.observe(document, {
      childList: true,
      subtree: true
    });
}
