from flask import Flask, render_template, request
from model import predict_crop

app = Flask(__name__)

FIELDS = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return {'status': 'ok'}


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            values = {f: float(request.form[f]) for f in FIELDS}
        except (KeyError, TypeError, ValueError):
            return render_template(
                'predict.html',
                error='⚠️ Invalid input! Please enter valid numbers in every field.',
            )

        result = predict_crop(
            values['n'], values['p'], values['k'], values['temperature'],
            values['humidity'], values['ph'], values['rainfall'],
        )
        return render_template('result.html', result=result, inputs=values)

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
