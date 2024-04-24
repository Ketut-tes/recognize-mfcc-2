import librosa
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


class mfccAlg:
    # untuk mengekstrak fitur dengan metode mfcc
    def extract_mfcc(self, y):
        mfccs = librosa.feature.mfcc(y=y, n_mfcc=13)
        normalized = scaler.fit_transform(mfccs)

        return normalized.T.flatten()

    # ekstrak dengan filepath
    def extract_mfcc_with_filepath(self, path):
        y, sr = librosa.load(path=path)
        mfccs = librosa.feature.mfcc(y=y, n_mfcc=13)
        normalized = scaler.fit_transform(mfccs)
        return normalized.T.flatten()
