import os
import pandas as pd
from split_audio import PrepocessData
from extract_feature import mfccAlg

# CREATE TRAIN MODEL

audioDirPath = "Song"

# List all audio files in the directory
audio_files = [os.path.join(audioDirPath, file) for file in os.listdir(
    audioDirPath) if file.endswith('.wav') or file.endswith('.mp3')]
print(len(audio_files))
for audioFile in audio_files:
    segmentLengthSec = 4

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

        pathToSave = f"Train_Model/{fileName}"
        os.makedirs(pathToSave, exist_ok=True)

        # Save train model
        csvPathTosave = pathToSave + f"/{fileName}.csv"
        df.to_csv(csvPathTosave, index=False)
    except Exception as e:
        print(e)

# YEAH
