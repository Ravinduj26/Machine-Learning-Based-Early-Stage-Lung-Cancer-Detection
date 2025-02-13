# Machine-Learning-Based-Early-Stage-Lung-Cancer-Detection

This project focuses on the development of an early-stage lung cancer detection system using machine learning techniques. It begins with an in-depth analysis of various lung cancer risk factors prevalent in Sri Lanka and then employs a predictive machine learning model to assess the risk of lung cancer in the general population.

**Overview**
Lung cancer continues to be one of the leading causes of death globally, and early detection is key to improving survival rates. This project leverages data on various risk factors, such as age, smoking habits, air pollution exposure, obesity, genetic risk, alcohol consumption, and lung disease history, to predict the likelihood of developing lung cancer.

The goal is to provide a user-friendly web interface where individuals can input their risk factors and receive a prediction of their lung cancer risk. The project uses a Random Forest model for prediction, which has been trained on a pre-processed dataset. The model is integrated with Flask to create a web-based application, making it easy for users to interact with the system and get results in real-time.

Features
Risk Factor Analysis: Inputs such as age, air pollution, alcohol usage, smoking habits, genetic risk, lung disease, and obesity are used to evaluate the risk.
Predictive Model: A machine learning model (Random Forest) is trained to predict the likelihood of lung cancer based on the provided risk factors.
LIME Explanation: The system explains the contribution of each risk factor to the final prediction using LIME (Local Interpretable Model-agnostic Explanations).
User Interface: A simple web interface built with Flask allows users to input their data and view predictions and explanations.
