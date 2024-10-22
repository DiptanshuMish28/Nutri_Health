import os
from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from ocr import preprocess_image, extract_text_from_image, extract_medical_fields

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads directory exists
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Load models
diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
# Load other models here as needed

# Function to calculate TDEE (Total Daily Energy Expenditure)
def calculate_tdee(activity_level, bodyweight, height, age, sex):
    if sex == 'male':
        BMR = 66 + (13.7 * bodyweight) + (5 * height) - (6.8 * age)
    else:  # female
        BMR = 655 + (9.6 * bodyweight) + (1.8 * height) - (4.7 * age)

    # Activity factor: 1.2 = sedentary, 1.55 = moderately active, 1.9 = very active
    if activity_level == 'low':
        TDEE = BMR * 1.2
    elif activity_level == 'medium':
        TDEE = BMR * 1.55
    else:  # high activity
        TDEE = BMR * 1.9
    return TDEE

# Function to calculate daily nutrition based on the goal
def recommend_diet(tdee, goal):
    if goal == 'cutting':
        energy = tdee - 300  # reduce by 300 calories for cutting
    elif goal == 'bulking':
        energy = tdee + 400  # increase by 400 calories for bulking
    else:
        energy = tdee  # maintain calories for maintenance

    # Macronutrient distribution based on Taiwan Ministry guidelines
    carbs = (energy * 0.5) / 4   # 50% of calories from carbs
    protein = (energy * 0.25) / 4  # 25% from protein
    fats = (energy * 0.25) / 9    # 25% from fats

    return energy, carbs, protein, fats

# Main prediction function for various diseases
def predict(values, dic):
    if len(values) == 8:
        # Diabetes prediction
        dic2 = {
            'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0,
            'NewBMI_Overweight': 0, 'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0,
            'NewGlucose_Low': 0, 'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0,
            'NewGlucose_Secret': 0
        }

        if dic['BMI'] <= 18.5:
            dic2['NewBMI_Underweight'] = 1
        elif 18.5 < dic['BMI'] <= 24.9:
            pass
        elif 24.9 < dic['BMI'] <= 29.9:
            dic2['NewBMI_Overweight'] = 1
        elif 29.9 < dic['BMI'] <= 34.9:
            dic2['NewBMI_Obesity 1'] = 1
        elif 34.9 < dic['BMI'] <= 39.9:
            dic2['NewBMI_Obesity 2'] = 1
        elif dic['BMI'] > 39.9:
            dic2['NewBMI_Obesity 3'] = 1

        if 16 <= dic['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        if dic['Glucose'] <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < dic['Glucose'] <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < dic['Glucose'] <= 126:
            dic2['NewGlucose_Overweight'] = 1
        elif dic['Glucose'] > 126:
            dic2['NewGlucose_Secret'] = 1

        dic.update(dic2)
        values2 = list(map(float, list(dic.values())))
        return diabetes_model.predict(np.asarray(values2).reshape(1, -1))[0]

    # Add other conditions (liver, heart, etc.) similarly as needed
    # ...

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    if request.method == 'POST':
        # Get form data for diabetes prediction
        pregnancies = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bloodpressure = int(request.form['BloodPressure'])
        skinthickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        # Collect additional data for diet recommendation
        weight = float(request.form['Weight'])  # in kg
        height = float(request.form['Height'])  # in cm
        activity_level = request.form['Activity']  # low, medium, high
        goal = request.form['Goal']  # cutting, maintaining, bulking
        sex = request.form['Sex']  # male, female

        # Predict diabetes
        input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)[0]

        if prediction == 0:
            result = 'The person is not diabetic'
        else:
            result = 'The person is diabetic'

        # Calculate TDEE and suggest diet
        tdee = calculate_tdee(activity_level, weight, height, age, sex)
        energy, carbs, protein, fats = recommend_diet(tdee, goal)

        # Display the prediction and diet recommendation
        return render_template('diabetes.html', prediction_text=result,
                               tdee_text=f"TDEE: {tdee:.2f} kcal",
                               diet_text=f"Recommended Diet: {energy:.2f} kcal/day (Carbs: {carbs:.2f}g, "
                                         f"Protein: {protein:.2f}g, Fats: {fats:.2f}g)")
    return render_template('diabetes.html')

# Other disease routes (liver, heart, etc.)
@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    data = None
    if request.method == 'POST':
        if 'image_file' in request.files:
            image_file = request.files['image_file']
            if image_file:
                image_path = os.path.join('uploads', image_file.filename)
                image_file.save(image_path)

                preprocessed_image = preprocess_image(image_path)
                if preprocessed_image is not None:
                    extracted_text = extract_text_from_image(preprocessed_image)
                    if extracted_text:
                        data = extract_medical_fields(extracted_text)
                        data['Gender'] = ''
                        not_found_fields = [field for field, value in data.items() if value == "Not found"]
                        if not_found_fields:
                            flash(f"Some fields could not be extracted: {', '.join(not_found_fields)}. Please fill them manually.", "warning")
                        else:
                            flash("Data successfully extracted from the image. Please review and correct if necessary.", "success")
                    else:
                        flash("Could not extract text from the image. Please enter the data manually.", "error")
                else:
                    flash("Could not process the image. Please try again with a clearer image.", "error")
                os.remove(image_path)
            else:
                flash("No file uploaded. Please select an image file.", "error")
    return render_template('liver.html', data=data)

# Add other disease prediction routes (heart, kidney, etc.)
# ...

if __name__ == '__main__':
    app.run(debug=True)
