import os
import pandas as pd
from split_audio import PrepocessData
from extract_feature import mfccAlg

# CREATE TRAIN MODEL
# Replace 'your_audio_file.wav' with the path to your audio file
audioFilePath = 'Song/Hindia-Cincin.wav'
# Segment length in seconds (4 seconds)
segmentLengthSec = 4

# Output folder path
output_folder = 'output_folder'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# initial pandas
df = pd.DataFrame()
pdata = PrepocessData()
mfccAg = mfccAlg()

# Split the audio file into segments
audio_segments, sr = pdata.split_audio(audioFilePath, segmentLengthSec)
print("LENGTH AUDIO SEGMES : ", len(audio_segments))

# file name yang digunakan untuk menyimpan data
fileName = os.path.splitext(os.path.basename(audioFilePath))[0]

# untuk menyimpan data untuk dijadikan csv
appendFinal = {}
for i in range(len(audio_segments)):
    result_mfcc = mfccAg.extract_mfcc(y=audio_segments[i])
    # print("MFCC : ", result_mfcc)
    # print("FILE NAME : ", fileName)
    feature_name = f"mfcc_{fileName}_{i}"
    appendFinal[feature_name] = result_mfcc
    # print(appendFinal)
    # df[feature_name] = result_mfcc

df = pd.DataFrame(appendFinal)

pathToSave = f"Train_Model/{fileName}"
os.makedirs(pathToSave, exist_ok=True)

# Save train model
csvPathTosave = pathToSave + f"/{fileName}.csv"
df.to_csv(csvPathTosave, index=False)

# first = audioFilePath.split(".")[0]
# firstName = first.split("/")[1]

# Export each segment to individual files in the output folder
# for i, segment in enumerate(audio_segments):
#     output_file_path = os.path.join(
#         output_folder, f"{firstName}_{i+1}.wav")
#     write(output_file_path, sr, segment)
