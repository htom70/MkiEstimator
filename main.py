import json
import sys
import pickle
import sklearn
import lightgbm
import flask
from flask import Flask, request, jsonify
import numpy as np
import time
import pandas as pd

# from sklearn.utils import column_or_1d
# from sklearn.decomposition import PCA


app = flask.Flask(__name__)
app.config["DEBUG"] = True


# api=Api(app)

class EstimatorContainer:
    def __init__(self):
        self.pipelineByIdCollection = dict()
        self.currencyEncoder = None
        self.countryEncoder = None

    def addPipelineById(self, id, pipeline):
        self.pipelineByIdCollection[id] = pipeline

    def createPrediction(self, input):
        requestParams = self.convertRawInput(input)
        complexResponse = list()
        pipelineByIdCollection = estimatorContainer.pipelineByIdCollection
        for pipelineId, currentFittedPipeline in pipelineByIdCollection.items():
            currentPrediction = currentFittedPipeline.predict(requestParams)[0]
            currentProbability = currentFittedPipeline.predict_proba(requestParams)[0]
            currentResponse = {
                'pipeLineId': pipelineId,
                'prediction': int(currentPrediction),
                'negativeProbability': float(currentProbability[0]),
                'positiveProbability': float(currentProbability[1])
            }
            complexResponse.append(currentResponse)
        result = jsonify(complexResponse)
        return result

    def convertRawInput(self, rawInput):
        cardNumberString = rawInput["card_number"]
        transactionType = rawInput["transaction_type"]
        amount = rawInput["amount"]
        currencyName = rawInput["currency_name"]
        responseCode = rawInput["response_code"]
        countryName = rawInput["country_name"]
        vendorCodeString = rawInput["vendor_code"]
        year = rawInput["year"]
        print(f"year: {year}")
        month = rawInput["month"]
        print(f"month: {month}")
        day = rawInput["day"]
        print(f"day: {day}")
        hour = rawInput["hour"]
        print(f"hour: {hour}")
        min = rawInput["min"]
        print(f"min: {min}")
        sec = rawInput["sec"]
        print(f"sec: {sec}")
        millis = rawInput["millis"]
        print(f"millisec: {millis}")
        cardNumber = int(cardNumberString)
        vendorCode = int(vendorCodeString)
        ts = pd.Timestamp(year, month, day, hour, min, sec, millis)
        julianDate = ts.to_julian_date()
        print(f"currencyName: {currencyName}")
        encodedCurrency = self.currencyEncoder.transform([currencyName])
        encodedCountry = self.countryEncoder.transform([countryName])
        requestParams = [
            [cardNumber, transactionType, julianDate, amount, encodedCurrency[0], responseCode, encodedCountry[0],
             vendorCode]]
        print(f"requestParams: {requestParams}")
        return requestParams


@app.route('/test', methods=['GET'])
def test():
    username = request.args.get('username')
    return jsonify('Hello ' + username)


@app.route('/init', methods=['POST'])
def init():
    content = request.get_json()
    ids = content.get('estimators')
    result = fillEstimatorContainer(ids)
    return jsonify(result)


@app.route('/api/v1/resources/predict_and_proba', methods=['POST'])
def api_predict_and_proba_sample():
    print("Post kérés start")
    start_time = time.process_time()
    content = request.get_json()
    # content = json.loads(content_json)
    # books.append(content)
    # rawInput = content.get("values")
    rawInput = content.get('values')
    print(rawInput)
    result = estimatorContainer.createPrediction(rawInput)
    return result

def fillEstimatorContainer(idCollection):
    for id in idCollection:
        currentPickledEstimatorName = 'estimator_' + str(id) + '.pickle'
        try:
            estimatorFile = open(f'c:/Users/Tom/Documents/MKI/estimators/python/{currentPickledEstimatorName}', 'rb')
            # estimatorFile = open(f'c:/Users/Tom/Documents/MKI/estimators/python/estimator_1.pickle',
            #                      'rb')
            currentEstimator = pickle.load(estimatorFile)
            fittedPipeline = currentEstimator.get("pipeline")
            estimatorContainer.addPipelineById(id, fittedPipeline)
            if estimatorContainer.countryEncoder is None:
                estimatorContainer.countryEncoder = currentEstimator.get("countryEncoder")
            if estimatorContainer.currencyEncoder is None:
                estimatorContainer.currencyEncoder = currentEstimator.get("currencyEncoder")
            estimatorFile.close()
        except FileNotFoundError:
            print(f'{currentPickledEstimatorName} not found')
        except IOError:
            print("IO Error")
    countryEncoder = estimatorContainer.countryEncoder
    print(countryEncoder.get_params())
    currencyEncoder = estimatorContainer.currencyEncoder
    print(currencyEncoder)
    for key, value in estimatorContainer.pipelineByIdCollection.items():
        print(f'Key: {key}: Value: {value}')
    return "OK"


if __name__ == '__main__':
    print("START")
    estimatorContainer = EstimatorContainer()
    portNumber = 8084
    app.run(port=portNumber)
