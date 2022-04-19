# Poetry Sonification
Poetry Sonification is a Python web app to generate sonifications of poems.
The project utilises a Deep Neural Network for music generation. The database
has been created from scraping (website listed in credits), and it consists of lyrics (sentences)
and notes pairs.

## System Requirements
- 64-bit versions of Microsoft Windows (or alternative Linux/MacOS).
- 8GB of RAM minimum.
- A CUDA-capable GPU.
- The NVIDIA CUDA Toolkit.
- Python 3.9 or above (including pip).
- A working version of JavaScript.
- An IDE (PyCharm is recommended).
- Substantial hard disk space (10GB or more).

## Installation
- Clone the repository:
    ```
    git clone 
    ```
- Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
- Run the app within the source folder:
    ```
    python -m flask run
    ```
- (Optional) Run the app from mobile (requires the device to be connected on the
  same network, and the network be private):
    ```
    python -m flask run --host 0.0.0.0
    ```
## Usage
The app is accessible at the following URL:
    ```
    localhost:5000/
    ```.
The user can decide whether to write a new text or upload a text file.
Upon submission, the app will generate a sonification of the text.
The music sheet can be visualised by pressing the apposite button (and so it can be downloaded).
It is also possible to download the generated result as a MIDI file.
Should the user be unable to generate a sonification, the app will output the appropriate error code.
It is also possible to upload the produced result to a database for further enhancement of the model.
This option is rendered by the rate button.

## Project Structure
The project is structured as follows:
- The source code is stored in the `flask_source` folder.
    - Certificates are stored in the `resources` folder.
    - HTML templates are stored in the `templates` folder.
    - The neural network model can be found in the `model.py` file.
    - The music generation is left to the `music_parser.py` file.
    - The main app is found in the `app.py` file.
    - Sonifications are stored at the path `static/assets/sonifications`.
    - Music sheets are stored at the path `static/assets/music-sheet-xml`.
- The tests are written in the `tests` folder.
- The music files used for evaluation are stored in the `evaluation` folder.
    


## Contributing
The project was developed as part of Alfredo Musumeci Undergraduate Dissertation at King's College London.
Pull requests are welcome. Substantial changes need open an issue first, detailing the changes exactly (along
with pro and cons). Make sure to include tests.

## Credits
- HTML Audio Player: `github.com/cifkao/html-midi-player/`
- OpenSheetMusicDisplay: `github.com/opensheetmusicdisplay/opensheetmusicdisplay`
- Music21: `web.mit.edu/music21/`
- Database: `noobnotes.net`
- SVG images: `svgrepo.com`

## License
[MIT License](https://opensource.org/licenses/MIT)
  