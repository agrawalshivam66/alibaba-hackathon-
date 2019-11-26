# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:32:43 2019

@author: Shivam-PC
"""

from flask import Flask, render_template, request
app = Flask(__name__)
from ReviewPredictor import predictKey

@app.route('/')
def index(methods=['GET', 'POST']):
   return render_template('Input.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
       
      text=request.form['Text']

      re = text
      writefile(re)

      return render_template("result.html",result = re)

def writefile(re):
    try:
    
      f = open("myfile.txt", "w")
      f.write(re)
      f.close()
    except:
        print('e')
        
    
if __name__ == '__main__':
    app.run()
    




    