import os
import pandas as pd
from split_audio import PrepocessData
from extract_feature import mfccAlg

# CREATE TRAIN MODEL

audioDirPath = "Song"

# List all audio files in the directory
audio_files = [os.path.join(audioDirPath, file) for file in os.listdir(
    audioDirPath) if file.endswith('.wav')]

audioFilePath = 'Song/Hindia-Rumah Ke Rumah.mp3'
# Segment length in seconds (4 seconds)
segmentLengthSec = 4

# initial pandas
df = pd.DataFrame()
pdata = PrepocessData()
mfccAg = mfccAlg()

# Split the audio file into segments
audio_segments, sr = pdata.split_audio(audioFilePath, segmentLengthSec)
# print("LENGTH AUDIO SEGMES : ", len(audio_segments))

# file name yang digunakan untuk menyimpan data
fileName = os.path.splitext(os.path.basename(audioFilePath))[0]

# untuk menyimpan data untuk dijadikan csv
appendFinal = {}
# print("LEN AUDIO SEGMENTS : ", len(audio_segments))
for i in range(len(audio_segments)):
    # print("PAPA : ", audio_segments[i])
    # print(f"LENGTH AUDIO SEGMENT - {i} : ", len(audio_segments[i]))
    result_mfcc = mfccAg.extract_mfcc(y=audio_segments[i])
    # print("MFCC : ", result_mfcc)
    # print("FILE NAME : ", fileName)
    feature_name = f"mfcc_{fileName}_{i}"
    appendFinal[feature_name] = result_mfcc
    # print(appendFinal)
    # df[feature_name] = result_mfcc
print(len(appendFinal))
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
