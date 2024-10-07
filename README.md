# Project Overview

This project compares the performance of different machine learning models, focusing on **Normal Regression** and **Random Forest**. The results are visualized in a graph where each line connects the corresponding values of **test MSE** and **test R²** for each experiment. The goal is to identify the model with the lowest MSE and highest R², which indicates better performance.

## How to Use

1. **Download the Project:**
   - Clone or download the repository from GitHub to your local machine.

2. **Docker Setup:**
   - Ensure Docker is running on your system.
   - Open the project folder in VSCode.

3. **Run Docker Compose:**
   - In VSCode, run the following command to start the Docker containers:
     ```bash
     docker-compose up
     ```
   - Wait for all dependencies to install.

4. **Check Container Status:**
   - Once the setup is complete, verify that the container is running by checking the Docker icon in your system tray.

5. **Run the Application:**
   - Open a terminal in the `source_code` folder.
   - Execute the following command to launch the app:
     ```streamlit run UI.py```
   - This will automatically open a new tab in your default internet browser.

6. **Test the Models:**
   - You are now ready to test both models (Normal Regression and Random Forest) using the provided interface.

7. **Boom! You're all set to compare the models**


![Screenshot of UI](A1App.png)
