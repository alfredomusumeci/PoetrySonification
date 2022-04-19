""" The flask application hosting the website and loading all components. """
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from model import RNNModel
import music_parser
import utils

app = Flask(__name__)
app.secret_key = 'f121ec36a0d2f52b8915c884dce8db5fa4d5344e4ed623ef'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MONGO_URI'] = 'mongodb+srv://cluster0.ai4gc.mongodb.net/myFirstDatabase?authSource=%24external' \
                          '&authMechanism=MONGODB-X509&retryWrites=true'

# Initialise time-consuming variables at loading time
sonification_model = RNNModel()
my_client = MongoClient(app.config['MONGO_URI'], tls=True, tlsCertificateKeyFile='flask_source/resources/X509-cert'
                                                                                 '-488296118074462419.pem')
my_db = my_client['sonifications']


@app.route('/generate_sonification', methods=['POST'])
def generate_sonification():
    """ Generate a sonification for the corresponding text grabbed via a POST request.
    :return: a response containing success, content, note and output path of the generated file. """

    global sonification_model

    # Delete the previous sonification and music sheet if any, based on the cookie id.
    cookie = request.cookies.get('song_id')
    if cookie is not None:
        utils.delete_file('flask_source/static/assets/sonifications/output' + cookie + '.mid')
        utils.delete_file('flask_source/static/assets/music-sheet-xml/output' + cookie + '.musicxml')

    # Get the text from the POST request and perform the sonification.
    content = request.form['content']
    notes = ''
    sonification_path_from_source = ''
    hexadecimalID = utils.generate_random_string(16)  # cookie ID

    try:
        prediction = sonification_model.predict(content)
        notes = prediction.split()
        sonification_path_from_root = music_parser.generate_song(notes, hexadecimalID)
        sonification_path_from_source = '/'.join(sonification_path_from_root.split('/')[1:])
        music_parser.generate_musicxml_file(notes, hexadecimalID)
        success = True
    except ValueError:  # Prediction has gone wrong, i.e., no known words in the vocabulary.
        success = False

    if not success:
        raise Exception("Impossible to generate a sonification for the given text.")

    response = jsonify({'success': success, 'content': content, 'notes': notes,
                        'result': sonification_path_from_source})
    response.set_cookie('song_id', hexadecimalID)
    return response


@app.route('/upload_to_db/', methods=['POST'])
def upload_to_db():
    """ Insert the sonification produced (text, notes) into the sonifications collection.
     :return: the id showing successful insertion. """

    text = request.values.get('text')
    notes = request.values.get('notes')

    sonification = {'text': text, 'sonification': notes}
    promise_result = my_db.sonifications.insert_one(sonification)

    return jsonify({'id': str(promise_result.inserted_id)})


@app.route('/about')
def about():
    """ Render the about page.
    :return: the rendered about page. """

    return render_template('about.html')


@app.route('/')
def index():
    """ Render the index page.
    :return: the rendered index page. """

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
