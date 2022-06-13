import pickle
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

def find_if_dropout(*args):
    result = loaded_model.predict(args)
    return result

print(find_if_dropout([56,1,4,4,3,2,10]))