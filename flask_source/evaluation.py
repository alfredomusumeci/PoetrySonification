""" A script to generate melodies using the baseline and the trained model. """
import baseline
import music_parser
import model
import utils

# From Tob Bodett.
happy_text1 = """
They say a person needs just three things to be truly happy in this world:
someone to love, something to do, and something to hope for.
"""

# From Stephen Chbosky.
happy_text2 = """
There's nothing like deep breaths after laughing that hard. Nothing in the 
world like a sore stomach for the right reasons.
"""

# From Walt Disney.
happy_text3 = """
Happiness is a state of mind. It's just according to the way you look at things.
"""

# From Albert Einstein.
happy_text4 = """
A calm and modest life brings more happiness than the pursuit of success combined 
with constant restlessness.
"""

# From Ellen Degeneres.
happy_text5 = """
The thing everyone should realize is that the key to happiness is being happy by 
yourself and for yourself.
"""

# From R. M. Drake.
sad_text1 = """
We must understand that sadness is an ocean, and sometimes we drown, 
while other days we are forced to swim.
"""

# From Nicholas Sparks.
sad_text2 = """
There are moments when I wish I could roll back the clock and take all 
the sadness away, but I have the feeling that if I did, the joy would be gone as well.
"""

# From Tupac Shakur.
sad_text3 = """
Behind every sweet smile, there is a bitter sadness that no one can ever see and feel.
"""

# From Carlos Del Valle.
sad_text4 = """
You are sad because you are wasting your potential. You know it, your family knows it, 
everyone knows it. Do something useful.
"""

# From Jonathan Safran Foer.
sad_text5 = """
You cannot protect yourself from sadness without protecting yourself from happiness.
"""

texts = [happy_text1, happy_text2, happy_text3, happy_text4, happy_text5,
         sad_text1, sad_text2, sad_text3, sad_text4, sad_text5]

if __name__ == '__main__':
    # Generate baseline predictions.
    baseline_sonifications = []
    for text in texts:
        baseline_sonifications.append(baseline.generate_random_melody(len(text)))

    for i, _ in enumerate(baseline_sonifications):
        output = '../evaluation/baseline'.format(id=i)
        music_parser.generate_song(_, utils.generate_random_string(16), output_filepath=output)

    print("Baseline Done")

    # Generate model predictions.
    sonification_model = model.RNNModel()
    sonification_model.build_models()
    model_sonifications = []

    for text in texts:
        prediction = sonification_model.predict(text)
        notes = prediction.split()
        model_sonifications.append(notes)

    for i, _ in enumerate(model_sonifications):
        output = '../evaluation/model'.format(id=i)
        music_parser.generate_song(_, utils.generate_random_string(16), output_filepath=output)

    print("Model Done")
