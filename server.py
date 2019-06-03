import tagger
import os
from flask import Flask, request, redirect, url_for, json, after_this_request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
THRESHOLD = 0.5

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['THRESHOLD'] = THRESHOLD

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            deepdanbooru_response = json.dumps(tagger.evaluate_post(os.path.join(
                app.config['UPLOAD_FOLDER'], filename), app.config['THRESHOLD'])),
            response = app.response_class(
                response=deepdanbooru_response,
                status=200,
                mimetype='application/json'
            )
            print("tags for " + filename + ":")
            print(deepdanbooru_response)
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print("removed " +
                      os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except OSError as e:
                print(e.strerror)
                print(e.code)
            return response
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

app.run(host="0.0.0.0", port=4443)
