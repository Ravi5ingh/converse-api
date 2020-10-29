import flask as fl
import utility.util as ut

app = fl.Flask(__name__)

# Load model
model_filename = __file__[0:__file__.rindex('\\')] + '\\..\\models\\model.pkl'
model = ut.read_pkl(model_filename)

@app.route('/say')
def say():

    text = fl.request.args.get('text', '')

    print(text)

    return model.predict(text)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()