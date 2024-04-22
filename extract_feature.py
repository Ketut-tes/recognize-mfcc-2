import librosa
from sklearn.preprocessing import MinMaxScaler


class mfccAlg:
    # untuk mengekstrak fitur dengan metode mfcc
    def extract_mfcc(y):
        mfccs = librosa.feature.mfcc(y=y, n_mfcc=13)
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(mfccs)

        return normalized.T.flatten()
