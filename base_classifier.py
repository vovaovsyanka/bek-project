from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os 
import re
import pickle

class MyLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, embedding_matrix):
        super(MyLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.2)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)

        lstm_out1, (hidden1, cell1) = self.lstm1(embedded)
        lstm_out2, (hidden2, cell2) = self.lstm2(lstm_out1)

        hidden_combined = torch.cat((hidden2[-2,:,:], hidden2[-1,:,:]), dim=1)

        x = self.relu(self.fc1(hidden_combined))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

def clean_and_tokenize(text):
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    tokens = text.split()
    return tokens

def clean_and_tokenize_for_tf_idf(text):
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

class BaseClassifier(ABC):
    """Абстрактный базовый класс для всех классификаторов"""
    
    @abstractmethod
    def classify(self, text: str) -> int:
        """
        Классифицирует текст
        Возвращает: 1 - вредоносный, 0 - безопасный
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Возвращает имя классификатора"""
        pass

class LSTM(BaseClassifier):
    def __load_lstm_model(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = MyLSTM(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_classes=checkpoint['num_classes'],
            embedding_matrix=checkpoint['embedding_matrix']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        word_index = checkpoint['word_index']
        label_encoder = checkpoint['label_encoder']
        vocab_size = checkpoint['vocab_size']
        embedding_dim = checkpoint['embedding_dim']
        hidden_dim = checkpoint['hidden_dim']
        num_classes = checkpoint['num_classes']
        embedding_matrix = checkpoint['embedding_matrix']
        max_sequence_length = checkpoint['max_sequence_length']
        
        return model, word_index, label_encoder, vocab_size, embedding_dim, hidden_dim, num_classes, embedding_matrix, max_sequence_length
    
    def __predict_text(self, text, model, word_index, label_encoder, max_length, device='cuda' if torch.cuda.is_available() else 'cpu'):
        tokens = clean_and_tokenize(text)
        
        sequence = [word_index.get(token, 0) for token in tokens]
        
        if len(sequence) < max_length:
            sequence = sequence + [0] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        
        input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1)
            confidence = torch.max(probabilities).item()
        
        predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())
        
        return predicted_label[0], confidence

    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("models", "security_lstm_pytorch.pth")
        self.model, self.word_index, self.label_encoder, self.vocab_size,\
        self.embedding_dim, self.hidden_dim, self.num_classes, self.embedding_matrix, self.max_sequence_length = self.__load_lstm_model(self.model_path)

    def classify(self, text: str) -> int:
        predicted_label, confidence = self.__predict_text(text, self.model, self.word_index, self.label_encoder, self.max_sequence_length)
        return 1 if predicted_label == "unsafe" else 0
    
    def get_name(self) -> str:
        return "LSTM"

class TF_IDF(BaseClassifier):
    
    def load_model(self,filename="model.pkl"):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        return model_data

    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("models", "tfidf_classifier.pkl")
        self.model_data = self.load_model(self.model_path)
        self.clf = self.model_data['model']
        self.vectorizer = self.model_data['vectorizer']
        self.label_encoder = self.model_data['label_encoder']



    def classify(self, text: str) -> int:
        query = text

        cleaned_query = clean_and_tokenize_for_tf_idf(query)
        query_tfidf = self.vectorizer.transform([cleaned_query]).toarray()
        y_pred = self.clf.predict(query_tfidf)
            
        predicted_class = self.label_encoder.inverse_transform([y_pred])[0]

        return 1 if str(predicted_class) == "unsafe" else 0
    
    def get_name(self) -> str:
        return "TF-IDF"

class Dictionary(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.file_dict_path = os.path.join("models", "banned_words.txt")

        with open(self.file_dict_path, encoding="utf-8") as file:
            all_words = file.read()

        self.list_of_words = all_words.split(", ")

    def classify(self, text: str) -> int:
        for word in self.list_of_words:
            if word in text:
                return 1
        return 0
    
    def get_name(self) -> str:
        return "Dictionary"

class RuBert(BaseClassifier):
    def __init__(self):
        super().__init__()

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join("models", "rubert")
        
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.__model = self.__model.to(self.__device)
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __preprocess_text(self, text, min_word_length=3):
        text = text.lower().strip()
        text = re.sub(r"\d", "", text)
        text = re.sub(r"[^a-zа-яё\s]", " ", text)
        words = [word for word in text.split() if len(word) >= min_word_length]
        text = " ".join(words)
        return text.strip()

    def __preprocess_function_single(self, example):
        cleaned_text = self.__preprocess_text(example, min_word_length=2)
        return self.__tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

    def classify(self, text: str) -> int:
        inputs = self.__preprocess_function_single(text)
        inputs = {k: v.to(self.__device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.__model(**inputs)
            prediction = outputs.logits.argmax(-1).item()
        return prediction
    
    def get_name(self) -> str:
        return "ruBert"