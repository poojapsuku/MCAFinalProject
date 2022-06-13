
#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os

from flask import Flask, redirect, url_for, render_template, request, flash
from load_model import find_if_dropout

app = Flask(__name__)
@app.route('/')
def home():
    return render_template("main.html")

@app.route('/res', methods = ['GET','POST'])
def my_form_post():
    ar_1 = request.form['income']
    ar_2 = request.form['disability']
    ar_3 = request.form['class']
    ar_4 = request.form['m1']
    ar_5 = request.form['m2']
    ar_6 = request.form['m3']
    ar_7 = request.form['attendence']

    ar = [[ar_1, ar_2, ar_3, ar_4, ar_5, ar_6, ar_7]]

    result = find_if_dropout(*ar)
    return render_template("res.html", r=result)

if __name__=="__main__":
    app.run(debug=True)