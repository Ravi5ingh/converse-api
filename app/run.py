import flask as fl

app = fl.Flask(__name__)

@app.route('/say')
def say():

    text = fl.request.args.get('text', '')

    return text

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()