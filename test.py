import pickle
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))


# income | disability | class | mark 1,2,3 | attendence

pooja = [[56,1,4,4,3,2,10],[27,0,4,4,2,4,12]]

tess = loaded_model.predict(pooja)

print(tess)