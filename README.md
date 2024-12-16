
# Customer Churn Prediction Using Machine Learning  

## Project Overview  
This project explores the use of **machine learning** to predict customer churn within the telecommunication industry. By identifying key drivers of churn, businesses can proactively develop strategies to retain at-risk customers.  

The main **research question** is:  
**How can machine learning help customer-focused companies reduce customer churn?**  

### Key Features  
- **Churn Prediction**: Predict whether a customer will churn using the CatBoostClassifier model.  
- **Insights**: Identify key churn drivers such as tenure, monthly charges, contract type, and payment method.  
- **Model Performance**: Achieves high recall, indicating strong performance in identifying at-risk customers.  
- **Deployment Ready**: Includes a simple Streamlit dashboard for demonstration purposes and FastAPI for integration.

---

## Results  
Key findings include:  
- Customers with **shorter tenure** and **higher monthly charges** are more likely to churn.  
- **Month-to-month contracts** increase churn risk compared to long-term contracts.  
- Manual payment methods (e.g., `electronic check`) correlate with higher churn.  

### Key Metrics:  
| Metric        | Score   |  
|---------------|---------|  
| Accuracy      | 77.93%  |  
| Recall        | 83.69%  |  
| Precision     | 55.60%  |  
| ROC-AUC       | 79.77%  |  

---

## Author  
**Oliver Gade Nielsen**  
- Cand.Merc Business Data Science 
- Aalborg University Business School  
