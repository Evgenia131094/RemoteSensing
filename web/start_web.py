from flask import Flask, render_template, request, url_for, flash, redirect
from components.start import start_action
from glob import glob
app = Flask(__name__)


@app.route('/', methods=['post', "get"])
def main_page():
    post = {"selected_mode": False}
    if request.method =='POST':
        mode = request.form['mode']
        saving_path, info_path = start_action(mode)
        post = {"selected_mode": True,
                "mode": mode,
                "saving_path": saving_path,
                "images": []}
        if info_path != "":
            post["images"] = glob(info_path)
    return render_template('main_page.html', post=post)


@app.route('/author')
def author():
    return render_template('author.html', selected_mode=False)


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')


@app.route('/training')
def training():
    return render_template('training.html')
