import joblib
model_file = 'spam_classifier_model.pkl'
vectorizer_file = 'vectorizer_fe.pkl'

model = joblib.load(model_file)

feature_extraction = joblib.load(vectorizer_file)


input_mail = ["""

"""]



def testing_model(input_mail):
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)

    if prediction[0] == 1:
        print('Ham mail')
    else: 
        print('spam')

testing_model(input_mail)



