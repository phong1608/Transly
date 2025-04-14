# app.py
from flask import Flask, render_template, request
from translate import Translate

app = Flask(__name__)
translate = Translate()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        language = request.form['language'] 
        
        if language == 'vi':
            translated_text = translate.en2vi(text)
        else:
            translated_text = translate.vi2en(text)
        
        return render_template('index.html', 
                             translated_text=translated_text,
                             selected_language=language)  
        
    return render_template('index.html', selected_language='en')  


if __name__ == '__main__':
    app.run(debug=True)