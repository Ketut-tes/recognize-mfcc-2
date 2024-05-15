import os
import pandas as pd
from split_audio import PrepocessData
from extract_feature import mfccAlg
import numpy as np


# CREATE TRAIN MODEL
def split_and_extract_mfcc(audioDirPath):
    # List all audio files in the directory
    audio_files = [os.path.join(audioDirPath, file) for file in os.listdir(
        audioDirPath) if file.endswith('.wav') or file.endswith('.mp3')]
    print(len(audio_files))
    for audioFile in audio_files:
        segmentLengthSec = 5

        # initial pandas
        df = pd.DataFrame()
        pdata = PrepocessData()
        mfccAg = mfccAlg()

        # Split the audio file into segments
        audio_segments, sr = pdata.split_audio(audioFile, segmentLengthSec)
        # print("LENGTH AUDIO SEGMES : ", len(audio_segments))
        # file name yang digunakan untuk menyimpan data
        fileName = os.path.splitext(os.path.basename(audioFile))[0]
        print("SONG : ", fileName)

        # untuk menyimpan data untuk dijadikan csv
        appendFinal = {}
        for i in range(len(audio_segments)):
            # print("LENGTH OF AUDIO SEGMENTS : ", len(audio_segments[i]))
            result_mfcc = mfccAg.extract_mfcc(y=audio_segments[i])
            # print("MFCC : ", result_mfcc)
            # print("FILE NAME : ", fileName)
            feature_name = f"mfcc_{fileName}_{i}"
            appendFinal[feature_name] = result_mfcc
            # print(appendFinal)
            # print("leng of mfcc : ", len(result_mfcc))
            # df[feature_name] = result_mfcc

        try:
            df = pd.DataFrame(appendFinal)

            pathToSave = "Train_Model"
            os.makedirs(pathToSave, exist_ok=True)

            # Save train model
            csvPathTosave = pathToSave + f"/{fileName}.csv"
            df.to_csv(csvPathTosave, index=False)
        except Exception as e:
            print(e)


def recognize(mfccToTest):
    pathTrain = "Train_Model"
    csvFile = [os.path.join(pathTrain, file) for file in os.listdir(
        pathTrain) if file.endswith('.csv')]

    dataAppend = []
    dataNameAppend = []
    for csV in csvFile:
        readCsv = pd.read_csv(csV)
        # header = readCsv.columns.tolist()
        # print(header)
        # body = pd.read_csv(csV, skiprows=1)
        # print(body)

        # print("to test : ", len(mfccToTest))
        for columnName, columnData in readCsv.items():
            # print(f"{columnName} panjang : {len(columnData)}")
            # TEST PAKE KNN
            dataAppend.append(columnData.T)
            dataNameAppend.append(columnName)

            # INI TEST RECOGNIZE DENGAN METODE BERBEDA
            # try:
            #     hasil = np.dot(
            #         columnData, mfccToTest) / (
            #             np.linalg.norm(
            #                 columnData) * np.linalg.norm(mfccToTest))
            #     print("Data", columnName, "- Hasil : ", hasil)
            # except Exception:
            #     print()
            #     # print(columnName)
            #     # print("ERROR : ", e)

    # COBA KNN
    k = 2
    Xtrain = dataAppend
    yTrain = dataNameAppend

    Xtest = mfccToTest
    print(Xtrain)
    distances = np.linalg.norm(Xtrain-Xtest, axis=1)

    nearestIndices = distances.argsort()[:k]

    nearestArray = [
        yTrain[nearestIndices[0]],
        yTrain[nearestIndices[1]],
                    ]

    print(nearestArray)

    return 0


audioDirPath = "Song"

# split_and_extract_mfcc(audioDirPath)


fileToTest = "uji_cincin.wav"
mffccAlg = mfccAlg()
mfccToTest = mffccAlg.extract_mfcc_with_filepath(path=fileToTest)
recognize(mfccToTest)
