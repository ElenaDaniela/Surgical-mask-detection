import librosa
import numpy as np

class features():
    def extract_features(file_path):  # functie care extrage caracteristicile fisierului adio
        try:
            audio, sr = librosa.load(file_path)  # res = "kaiser_best" este folosit ca default
            mfccs_nescalat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_scalat = np.mean(mfccs_nescalat.T, axis=0)

        except Exception as e:
            print("Eroare la: ", file_path)
            return None

        return mfccs_scalat