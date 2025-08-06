from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_hd():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['chestpaintype'])
        trestbps = int(request.form['restingbp'])
        chol = int(request.form['cholesterol'])
        fbs = int(request.form['fastingbs'])
        restecg = int(request.form['restingecg'])
        thalach = int(request.form['maxhr'])
        exang = int(request.form['exerciseangina'])
        oldpeak = float(request.form['oldpeak'])
        stslope = int(request.form['stslope'])

        input_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,stslope]

        data = pd.read_csv(os.path.join('static', 'files', 'heart.csv'))
        le = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = le.fit_transform(data[column])

        x = data.drop('HeartDisease', axis=1)
        y = data['HeartDisease']

        #Split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        #Model
        model = Sequential([
            Dense(64, activation='relu', input_shape = (x_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        #Compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        #Train
        model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))

        input_df = pd.DataFrame([input_data], columns=x.columns)

        for column in input_df.columns:
            if input_df[column].dtype == 'object':
                input_df[column] = le.transform(input_df[column])
                
        prediction = model.predict(input_df)
        accuracy = accuracy_score(y_test, (model.predict(x_test) > 0.5).astype(int))

        return render_template('home.html', result='Heart Disease Detected' if prediction[0][0] == 1 else 'No Heart Disease Detected', accuracy = round(accuracy * 100, 2))
    return render_template('home.html', result='', accuracy='')

if __name__ == '__main__':
    app.run(debug=True, port=3000)