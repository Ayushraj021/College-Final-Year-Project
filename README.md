# Disease Prediction Web Application

This project is a web-based application for predicting diseases based on user-reported symptoms using machine learning models. The application leverages three different models (Support Vector Machine, Naive Bayes, and Random Forest) to provide a prediction based on the symptoms entered by the user.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [License](#license)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/disease-prediction.git
    cd disease-prediction
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the trained models and data files:**
   - Ensure `final_svm_model.pkl`, `final_nb_model.pkl`, and `final_rf_model.pkl` are in the project directory.
   - Place `Training.csv` in the project directory.

4. **Run the Flask application:**
    ```bash
    python app.py
    ```

5. **Access the web application:**
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

- Navigate to the home page to see the welcome message.
- Go to the prediction page to enter symptoms and get a disease prediction.

## Project Structure

```
disease-prediction/
│
├── templates/
│   ├── index.html          # Home page template
│   └── prediction.html     # Prediction page template
│
├── app.py                  # Main Flask application
├── Training.csv            # Training data
├── final_svm_model.pkl     # Trained SVM model
├── final_nb_model.pkl      # Trained Naive Bayes model
├── final_rf_model.pkl      # Trained Random Forest model
├── encoder_classes.npy     # Encoder classes file
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## How It Works

1. **Data Loading and Preprocessing:**
   - The `Training.csv` file is loaded, and unnecessary columns are dropped.
   - Symptoms are extracted from the dataset and converted to a format suitable for the model.

2. **Model Loading:**
   - The pre-trained models (`final_svm_model.pkl`, `final_nb_model.pkl`, `final_rf_model.pkl`) are loaded using `joblib`.
   - The label encoder is fitted on the prognosis column of the data to handle categorical labels.

3. **Prediction Logic:**
   - User inputs symptoms via a web form.
   - The input symptoms are processed and converted into a format that the models can understand.
   - Each model makes a prediction, and the final prediction is determined by the most frequent prediction among the models.

4. **Web Interface:**
   - The web application has two main pages: a home page and a prediction page.
   - Users can enter their symptoms on the prediction page and receive a predicted disease.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to reach out if you have any questions or need further assistance!

---

**Note:** Ensure the models and the training data file are placed in the correct directories as specified in the project structure for the application to work correctly.
