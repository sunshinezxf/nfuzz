# this is the entry point for nfuzz

from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


# load the program under test
@app.route('/instance/load')
def load_instance():
    return


# load test input
@app.route('/tinput/load')
def load_tinput():
    return


# load test case
@app.route('/tcase/load')
def load_tcase():
    return


# config execution params
@app.route('/terminate/config')
def config_termination():
    return


# run fuzzing
@app.route('/nfuzz/execute')
def execute():
    return


if __name__ == '__main__':
    app.run()
