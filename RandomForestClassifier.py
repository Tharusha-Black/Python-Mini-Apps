{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a911583d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: librosa in c:\\users\\tharu\\anaconda3\\lib\\site-packages (0.10.1)\n",
      "Requirement already satisfied: audioread>=2.1.9 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (3.0.1)\n",
      "Requirement already satisfied: numpy!=1.22.0,!=1.22.1,!=1.22.2,>=1.20.3 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (1.24.3)\n",
      "Requirement already satisfied: scipy>=1.2.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (1.11.1)\n",
      "Requirement already satisfied: scikit-learn>=0.20.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (1.3.0)\n",
      "Requirement already satisfied: joblib>=0.14 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (1.2.0)\n",
      "Requirement already satisfied: decorator>=4.3.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (5.1.1)\n",
      "Requirement already satisfied: numba>=0.51.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (0.57.1)\n",
      "Requirement already satisfied: soundfile>=0.12.1 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (0.12.1)\n",
      "Requirement already satisfied: pooch>=1.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (1.8.1)\n",
      "Requirement already satisfied: soxr>=0.3.2 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (0.3.7)\n",
      "Requirement already satisfied: typing-extensions>=4.1.1 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (4.7.1)\n",
      "Requirement already satisfied: lazy-loader>=0.1 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (0.2)\n",
      "Requirement already satisfied: msgpack>=1.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from librosa) (1.0.3)\n",
      "Requirement already satisfied: llvmlite<0.41,>=0.40.0dev0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from numba>=0.51.0->librosa) (0.40.0)\n",
      "Requirement already satisfied: platformdirs>=2.5.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from pooch>=1.0->librosa) (3.10.0)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from pooch>=1.0->librosa) (23.1)\n",
      "Requirement already satisfied: requests>=2.19.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from pooch>=1.0->librosa) (2.31.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from scikit-learn>=0.20.0->librosa) (2.2.0)\n",
      "Requirement already satisfied: cffi>=1.0 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from soundfile>=0.12.1->librosa) (1.15.1)\n",
      "Requirement already satisfied: pycparser in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from cffi>=1.0->soundfile>=0.12.1->librosa) (2.21)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (1.26.16)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\tharu\\anaconda3\\lib\\site-packages (from requests>=2.19.0->pooch>=1.0->librosa) (2023.7.22)\n"
     ]
    }
   ],
   "source": [
    "!pip install librosa"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d40578e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import librosa\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b764f1e0",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import librosa\n",
    "import numpy as np\n",
    "\n",
    "# Define labels\n",
    "labels = ['breed1', 'breed2']  # Replace with actual breed names\n",
    "\n",
    "# Initialize lists to store audio data and labels\n",
    "data = []\n",
    "target = []\n",
    "\n",
    "# Set a fixed length for audio data\n",
    "fixed_length = 17240  \n",
    "\n",
    "# Iterate through folders\n",
    "dataset_path = 'dog_bark_dataset'\n",
    "for label_idx, label in enumerate(labels):\n",
    "    folder_path = os.path.join(dataset_path, label)\n",
    "    for filename in os.listdir(folder_path):\n",
    "        if filename.endswith('.wav'):\n",
    "            # Load audio file\n",
    "            audio_path = os.path.join(folder_path, filename)\n",
    "            audio_data, _ = librosa.load(audio_path, sr=None, duration=10.0)  # Limit duration to 10 second\n",
    "            # If audio data length is less than fixed length, pad with zeros\n",
    "            if len(audio_data) < fixed_length:\n",
    "                audio_data = np.pad(audio_data, (0, fixed_length - len(audio_data)))\n",
    "            # If audio data length is greater than fixed length, trim\n",
    "            elif len(audio_data) > fixed_length:\n",
    "                audio_data = audio_data[:fixed_length]\n",
    "            \n",
    "            # Append audio data and label\n",
    "            data.append(audio_data)\n",
    "            target.append(label_idx)\n",
    "\n",
    "# Convert lists to numpy arrays\n",
    "data = np.array(data)\n",
    "target = np.array(target)\n",
    "\n",
    "# Optionally, save the data and labels to a file\n",
    "np.save('audio_data.npy', data)\n",
    "np.save('audio_labels.npy', target)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8bad64b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.7627118644067796\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Load the data\n",
    "data = np.load('audio_data.npy')\n",
    "labels = np.load('audio_labels.npy')\n",
    "\n",
    "# Split the dataset into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)\n",
    "\n",
    "# Preprocess the data (if necessary)\n",
    "# Example: Normalize the data\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Define and train the model\n",
    "model = RandomForestClassifier()  # Example classifier (replace with your chosen model)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate the model\n",
    "accuracy = model.score(X_test, y_test)\n",
    "print(\"Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "fb5ac35f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['dog_breed_detection.joblib']"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Assuming your model is named 'model'\n",
    "joblib.dump(model, 'dog_breed_detection.joblib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "082a2c5b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted breed: breed1\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import librosa\n",
    "import joblib\n",
    "\n",
    "# Load the trained model using joblib\n",
    "model = joblib.load('dog_breed_detection.joblib')\n",
    "\n",
    "# Preprocess the new audio file\n",
    "def preprocess_audio(audio_file):\n",
    "    # Example: Extract features from the audio file using librosa\n",
    "    features = extract_features(audio_file)\n",
    "    # Example: Normalize the features\n",
    "    scaler = StandardScaler()\n",
    "    features = scaler.fit_transform(features)\n",
    "    return features\n",
    "\n",
    "# Function to extract features from the audio file\n",
    "def extract_features(audio_file):\n",
    "    # Example: Use librosa to load and extract features from the audio file\n",
    "    # Replace this with your own feature extraction code\n",
    "    features = librosa.feature.mfcc(y=audio_file)  # Adjust n_mfcc as needed\n",
    "    return features\n",
    "\n",
    "breed_names = {\n",
    "    0: 'breed1',\n",
    "    1: 'breed2'\n",
    "    # Add more mappings if you have additional breeds\n",
    "}\n",
    "\n",
    "# Function to predict the breed name\n",
    "def predict_breed_name(audio_file):\n",
    "    # Preprocess the audio file\n",
    "    processed_audio = preprocess_audio(audio_file)\n",
    "    # Reshape the processed audio to match the model input shape\n",
    "    processed_audio = np.reshape(processed_audio, (1, -1))  # Reshape to 2D array\n",
    "    # Make predictions using the trained model\n",
    "    predicted_label = model.predict(processed_audio)\n",
    "    # Get the breed name corresponding to the predicted index\n",
    "    predicted_breed_name = breed_names.get(predicted_label[0], 'Unknown')\n",
    "    return predicted_breed_name\n",
    "\n",
    "def load_audio(audio_file_path):\n",
    "    audio_data, s = librosa.load(audio_file_path, sr=44100)\n",
    "    return audio_data\n",
    "\n",
    "audio_file = load_audio('dog_bark_dataset/breed2/audio55.wav')\n",
    "predicted_breed_name = predict_breed_name(audio_file)\n",
    "print(\"Predicted breed:\", predicted_breed_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3922311d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
