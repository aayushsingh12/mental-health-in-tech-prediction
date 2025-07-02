import gradio as gr
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("model/mental_health_model.pkl", "rb"))
transformer = pickle.load(open("model/transformer.pkl", "rb"))
encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

def predict(*inputs):
    input_dict = dict(zip([
        'self_employed', 'work_interfere', 'Age', 'Gender', 'family_history', 'no_employees', 'remote_work', 'tech_company', 'benefits',
        'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
        'mental_health_consequence', 'phys_health_consequence', 'coworkers',
        'supervisor', 'mental_health_interview', 'phys_health_interview',
        'mental_vs_physical', 'obs_consequence'
    ], inputs))
    
    X = pd.DataFrame([input_dict])  # wrap in list to keep 2D
    X_transformed = transformer.transform(X)
    pred = model.predict(X_transformed)
    return encoder.inverse_transform(pred)[0]


input_components=[
    gr.Dropdown(["Yes", "No"], label="Are you self-employed?"),
    gr.Dropdown(["Often", "Rarely", "Sometimes", "Never"], label="If you have a mental health condition, do you feel that it interferes with your work?"),
    gr.Slider(15, 80, step=1, label="Age"),
    gr.Dropdown(["Male", "Female", "Other"], label="Gender"),
    gr.Dropdown(["Yes", "No"], label="Do you have a family history of mental illness?"),
    gr.Dropdown(["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"], label="How many employees does your company or organization have?"),
    gr.Dropdown(["Yes", "No"], label="Do you work remotely (outside of an office) at least 50% of the time?"),
    gr.Dropdown(["Yes", "No"], label="Is your employer primarily a tech company/organization?"),
    gr.Dropdown(["Yes", "No", "Don't know"], label="Does your employer provide mental health benefits?"),
    gr.Dropdown(["Yes", "No", "Not sure"], label="Do you know the options for mental health care your employer provides?"),
    gr.Dropdown(["Yes", "No", "Don't know"], label="Has your employer ever discussed mental health as part of an employee wellness program?"),
    gr.Dropdown(["Yes", "No", "Don't know"], label="Does your employer provide resources to learn more about mental health issues and how to seek help?"),
    gr.Dropdown(["Yes", "No", "Don't know"], label="Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?"),
    gr.Dropdown(["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"], label="How easy is it for you to take medical leave for a mental health condition?"),
    gr.Dropdown(["Yes", "No", "Maybe"], label="Do you think that discussing a mental health issue with your employer would have negative consequences?"),
    gr.Dropdown(["Yes", "No", "Maybe"], label="Do you think that discussing a physical health issue with your employer would have negative consequences?"),
    gr.Dropdown(["Yes", "No", "Some of them"], label="Would you be willing to discuss a mental health issue with your coworkers?"),
    gr.Dropdown(["Yes", "No", "Some of them"], label="Comfortable with supervisor?"),
    gr.Dropdown(["Yes", "No"], label="Would you bring up a mental health issue with a potential employer in an interview?"),
    gr.Dropdown(["Yes", "No"], label="Would you bring up a physical health issue with a potential employer in an interview?"),
    gr.Dropdown(["Yes", "No", "Don't know"], label="Do you feel that your employer takes mental health as seriously as physical health?"),
    gr.Dropdown(["Yes", "No"], label="Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?")
]

demo=gr.Interface(
    fn=predict,
    inputs=input_components,
    outputs=gr.Textbox(label="Prediction: Will the individual seek mental health treatment?"),
    title="Mental Health Treatment Predictor"
)

demo.launch()
