from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and its expected columns
model = joblib.load('student_score_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input and convert to dictionary
    input_data = {
        'Hours_Studied': int(request.form['Hours_Studied']),
        'Attendance': int(request.form['Attendance']),
        'Parental_Involvement': request.form['Parental_Involvement'],
        'Access_to_Resources': request.form['Access_to_Resources'],
        'Extracurricular_Activities': request.form['Extracurricular_Activities'],
        'Sleep_Hours': int(request.form['Sleep_Hours']),
        'Previous_Scores': int(request.form['Previous_Scores']),
        'Motivation_Level': request.form['Motivation_Level'],
        'Internet_Access': request.form['Internet_Access'],
        'Tutoring_Sessions': int(request.form['Tutoring_Sessions']),
        'Family_Income': request.form['Family_Income'],
        'Teacher_Quality': request.form['Teacher_Quality'],
        'School_Type': request.form['School_Type'],
        'Peer_Influence': request.form['Peer_Influence'],
        'Physical_Activity': int(request.form['Physical_Activity']),
        'Learning_Disabilities': request.form['Learning_Disabilities'],
        'Parental_Education_Level': request.form['Parental_Education_Level'],
        'Distance_from_Home': request.form['Distance_from_Home'],
        'Gender': request.form['Gender']
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encode and align columns
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict the score
    prediction = model.predict(df_encoded)[0]

    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
