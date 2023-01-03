import os
import pickle
import pandas as pd
import requests
import xgboost as xgb
from flask import Flask, request, Response


#collecting the past archive and imnport class
from insurance.Insurance import Insurance


model = xgb.XGBClassifier()

model.load_model('/home/guilherme/Documentos/repos/pa_health_cross_sell/projeto/model/model_cross_sell.json')

#start api
app = Flask(__name__)

#creating url (endpoint) to send data
@app.route('/predict', methods=['POST'] )

def insurance_predict():
    test_json = request.get_json()
    
    if test_json: #In there is data
        
        if isinstance(test_json, dict): #unique example
            #in case there is only row
            test_row = pd.DataFrame(test_json, index=[0] )
            
        else:
            #colletcting all json of all all rows
            test_row= pd.DataFrame(test_json, columns=test_json[0].keys() )#multiples examples
        
        #Instance the class (Making the copy)
        
        pipeline = Insurance()
        
        df1 = pipeline.cleaning_data(test_row)
        print('df1 done')
        
        df2 = pipeline.feature_engineering(df1)
                      
        df3 = pipeline.data_preparation(df2)
        
        #predict of probability
        df_response= pipeline.get_prediction(model, test_row, df3)
       
        return df_response
    
    else: #If case empty
        return Response( '{}', status=200, mimetype='application/json')
    
    
if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host = '0.0.0.0', port=port )


