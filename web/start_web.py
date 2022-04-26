from flask import Flask, render_template, request, url_for, flash, redirect
from components.start import start_action

app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template('main_page.html')


@app.route('/mode/<mode>')
def main_mode_page(mode):
    saving_path, info_path = start_action(mode=mode)
    return render_template('main_page.html', selected_mode=True, mode=mode, saving_path=saving_path, info_path=info_path)


@app.route('/author')
def author():
    return render_template('author.html', selected_mode=False)


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')


@app.route('/training')
def training():
    return render_template('training.html')
