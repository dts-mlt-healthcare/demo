import numpy as np

#one hot encoding symtomps
def onehot_sym(symptoms, symps_data):
    matrix_sym = [0] * len(symps_data) #inisiate onehot matrix
    for symptom in symptoms:
        try:
            index = symps_data.index(symptom)
            matrix_sym[index] = 1
        except:pass # pass, when the symptom not in symps_data
    return np.array(matrix_sym).reshape(1,-1)

def clean(symptoms):
    sym = symptoms.split(',')
    sym = [s.strip() for s in sym]
    return sym
    
def desc_dis(result, desc_data):
    desc = desc_data.Description[desc_data.Disease == result].tolist()[0]
    return desc

def prec_dis(result, prec_data):
    prec = prec_data[['Precaution_1','Precaution_2','Precaution_3','Precaution_4']][prec_data.Disease == result]
    prec = prec.values.tolist()[0]
    return prec

def pred_disease(symps, desc_data, prec_data, model, enc, symps_data):
    # symps = clean(symps)
    symps = onehot_sym(symps, symps_data)
    result = enc.inverse_transform(model.predict(symps))[0][0]
    proba =  np.max(model.predict(symps))
    desc = desc_dis(result, desc_data)
    prec = prec_dis(result, prec_data)
    return result, proba, desc, prec