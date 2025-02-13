# Machine-Learning-Based-Early-Stage-Lung-Cancer-Detection

This project focuses on the development of an early-stage lung cancer detection system using machine learning techniques. It begins with an in-depth analysis of various lung cancer risk factors prevalent in Sri Lanka and then employs a predictive machine learning model to assess the risk of lung cancer in the general population.


**Overview**

Lung cancer continues to be one of the leading causes of death globally, and early detection is key to improving survival rates. This project leverages data on various risk factors, such as age, smoking habits, air pollution exposure, obesity, genetic risk, alcohol consumption, and lung disease history, to predict the likelihood of developing lung cancer.

The goal is to provide a user-friendly web interface where individuals can input their risk factors and receive a prediction of their lung cancer risk. The project uses a Random Forest model for prediction, which has been trained on a pre-processed dataset. The model is integrated with Flask to create a web-based application, making it easy for users to interact with the system and get results in real-time.


**Features**

**Risk Factor Analysis:** Inputs such as age, air pollution, alcohol usage, smoking habits, genetic risk, lung disease, and obesity are used to evaluate the risk.
**Predictive Model:** A machine learning model (Random Forest) is trained to predict the likelihood of lung cancer based on the provided risk factors.
**LIME Explanation:** The system explains the contribution of each risk factor to the final prediction using LIME (Local Interpretable Model-agnostic Explanations).
**User Interface:** A simple web interface built with Flask allows users to input their data and view predictions and explanations.


**Project Structure**

![image](https://github.com/user-attachments/assets/d8226463-5b0e-466f-959d-351563353bfb)


**Installation**

**Prerequisites**

1. Python 3.x
2. Flask
3. Other dependencies (listed in requirements.txt)


**Installation Steps**

**1. Clone the Repository:**

git clone https://github.com/your-username/Machine-Learning-Based-Early-Stage-Lung-Cancer-Detection.git
cd Machine-Learning-Based-Early-Stage-Lung-Cancer-Detection

**2. Create a Virtual Environment (optional but recommended):**

python -m venv venv

**3. Install the required dependencies:**

pip install -r requirements.txt

**4. Download or Train the Model:**

If you already have the trained model Random Forest_model.pkl, place it in the root directory of the project. Alternatively, you can train the model using Jupyter notebooks on the provided dataset. (Details on training can be found in the notebooks section of the repository.)

**5. Run the Flask Application:**

python app.py
The application will run locally at http://127.0.0.1:5000/.


**Model Overview**

The core of the application is a Random Forest model trained to predict the likelihood of lung cancer based on multiple risk factors. These factors include:

Age: The user's age.

Air Pollution: Level of exposure to air pollution.

Alcohol Usage: The user's alcohol consumption habits.

Genetic Risk: The genetic predisposition to lung cancer.

Lung Disease: Pre-existing lung conditions.

Obesity: The user's weight status.

Smoking: The user's smoking habits.


**Model Training**

The model is trained on a pre-processed dataset that contains historical information related to lung cancer risk factors. The data has been cleaned and preprocessed to ensure quality and consistency.


**Model Prediction**

When the user enters their data, the Flask app uses the trained model to make a prediction. The model outputs a severity class (e.g., Healthy, Low, Medium, High), indicating the user's risk level for developing lung cancer.


**LIME Explanation**

The Local Interpretable Model-agnostic Explanations (LIME) library is used to generate human-readable explanations of the model's predictions. LIME helps the user understand which risk factors contributed the most to their predicted risk.


**Usage**

Step 1: Enter the Data
Navigate to the web interface in your browser (http://127.0.0.1:5000/) and enter the following details:

Age

Air Pollution Exposure

Alcohol Usage

Genetic Risk

Lung Disease History

Obesity Status

Smoking Habits


**Step 2: Get Prediction**

After submitting the data, the system will provide:

Prediction: The risk level for lung cancer (Healthy, Low, Medium, High).

Explanation: The contribution of each risk factor to the prediction, generated using LIME.


**Example Output:**
Prediction: High Risk of Lung Cancer
Explanation:
Smoking contributes to an increase in risk by 30%.
Age contributes to an increase in risk by 20%.
Obesity contributes to a decrease in risk by 5%.


**Dependencies**

The required dependencies for this project are listed in the requirements.txt file:

Flask
Numpy
Pandas
Scikit-learn
LIME
Joblib
Pickle


**Install them via:**

pip install -r requirements.txt
Contributing
Contributions are welcome! If you'd like to contribute to the project, please fork the repository, create a new branch, and submit a pull request with your changes.
