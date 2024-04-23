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

    def split_audio(self, input_file, segment_duration=4):
        # Load the audio file
        y, sr = librosa.load(input_file, sr=None)

        # Calculate the number of samples in each segment
        segment_samples = int(segment_duration * sr)

        # Calculate the number of segments
        num_segments = len(y) // segment_samples

        segments = []
        # Iterate over each segment
        for i in range(num_segments):
            start_sample = i * segment_samples
            end_sample = start_sample + segment_samples

            # Extract the segment
            segment = y[start_sample:end_sample]
            segments.append(segment)
        return segments, sr

    # split audio dengan setiap detik memiliki 4 detik durasi
    # def split_audio(self, audioFilePath, segmentLengthSec):
    #     # Load the audio file
    #     y, sr = librosa.load(audioFilePath, sr=None)
    #     print("y : ", len(y))

    #     # Calculate segment length in samples
    #     customLengthSamples = int(1 * sr)
    #     segmentLengthSamples = int(segmentLengthSec * sr)
    #     # print("Segment length samples : ", segmentLengthSamples)

    #     # Split the audio into segments
    #     segments = []
    #     for i in range(0, len(y), customLengthSamples):
    #         print("i :", i)
    #         print("segment : ", i+segmentLengthSec)
    #         segment = y[i:i+segmentLengthSamples]
    #         # print("Segment : ", segment)
    #         # print("CHECK ALL ZEROS : ", check_all_zeros(segment))
    #         if not self.check_all_zeros(segment):
    #             segments.append(segment)
    #     return segments, sr  # Return the sampling rate as well
