import librosa
# from scipy.io.wavfile import write


class PrepocessData(object):
    # mengecek jika semua value dari array itu 0 atau tidak
    def check_all_zeros(self, arr):
        """ini untuk mengecek apakah setiap array value tersebut
        bernilai 0 atau tidak
        jika 0 semua maka return true
        jika tidak return false"""
        return all(item == 0 for item in arr)

    # split audio dengan setiap detik memiliki 4 detik durasi
    def split_audio(self, audioFilePath, segmentLengthSec):
        # Load the audio file
        y, sr = librosa.load(audioFilePath, sr=None)
        print("y : ", len(y))

        # Calculate segment length in samples
        customLengthSamples = int(1 * sr)
        segmentLengthSamples = int(segmentLengthSec * sr)
        # print("Segment length samples : ", segmentLengthSamples)

        # Split the audio into segments
        segments = []
        for i in range(0, len(y), customLengthSamples):
            # print("i :", i)
            # print("segment : ", i+segmentLengthSec)
            segment = y[i:i+segmentLengthSamples]
            # print("Segment : ", segment)
            # print("CHECK ALL ZEROS : ", check_all_zeros(segment))
            if not self.check_all_zeros(segment):
                segments.append(segment)
        return segments, sr  # Return the sampling rate as well
