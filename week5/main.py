import pickle

with open('model1.bin','rb') as f_out:
    model= pickle.load(f_out)

with open('dv.bin','rb') as f_out:
    dv= pickle.load(f_out)

#test = {"job": "management", "duration": 400, "poutcome": "success"}
#client = {"job": "student", "duration": 280, "poutcome": "failure"}

client ={"job": "management", "duration": 400, "poutcome": "success"}
X = dv.transform([client])
print(X.shape)
print(model.predict_proba(X)[0,1])
