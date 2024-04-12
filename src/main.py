#Importing the necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from streamlit_option_menu import option_menu
from plotting_helpers import (plot_top_5_products, plot_top_5_issues, plot_top_5_issues_in_product, plot_top_10_companies_complaints,
                              plot_top_10_states_most_complaints, plot_top_10_states_least_complaints, complaints_by_year, 
                              complaints_across_states)
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle
import time
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Setting page config
st.set_page_config(page_title='CFPB Consumer Complaint Insights', page_icon='📋',
                    layout="wide", initial_sidebar_state='expanded')

@st.cache_data(show_spinner=False)
def load_process_data():
    df = pd.read_csv('../complaints.csv')
    df['Date received'] = pd.to_datetime(df['Date received'])

    cols_to_consider = ['Product','Sub-product','Issue','Sub-issue','Consumer complaint narrative','Company public response','Company',
                        'State', 'ZIP code', 'Date received']
    df_new = df[cols_to_consider]

    df_new = df_new.dropna()

    product_map = {'Credit reporting or other personal consumer reports' : 'Credit Reporting',
                'Credit reporting, credit repair services, or other personal consumer reports' : 'Credit Reporting',
                'Payday loan, title loan, personal loan, or advance loan' : 'Loans / Mortgage',
                'Payday loan, title loan, or personal loan' : 'Loans / Mortgage',
                'Student loan' : 'Loans / Mortgage',
                'Vehicle loan or lease' : 'Loans / Mortgage',
                'Debt collection' : 'Debt collection',
                'Credit card or prepaid card' : 'Credit/Prepaid Card',
                'Credit card' : 'Credit/Prepaid Card',
                'Prepaid card' : 'Credit/Prepaid Card',
                'Mortgage' : 'Loans / Mortgage',
                'Checking or savings account' : 'Checking or savings account'  
                }

    df_new.loc[:,'Product'] = df_new['Product'].map(product_map)


    df_new['complaint length'] = df_new['Consumer complaint narrative'].apply(lambda x : len(x))
    df_new = df_new[df_new['complaint length'] > 20]

    complaints_to_exclude = ['See document attached', 'See the attached documents.', 'Incorrect information on my credit report', 'incorrect information on my credit report',
    'please see attached file','Please see documents Attached','Incorrect information on my credit report.', 'Please see attached file', 'see attached',
    'See attached', 'SEE ATTACHED DOCUMENTS', 'See Attached', 'SEE ATTACHMENT', 'SEE ATTACHMENTS', 
    'XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX']

    df_new = df_new[~df_new['Consumer complaint narrative'].isin(complaints_to_exclude)]

    return df_new

# Load the processed data
df = load_process_data()

# Loading the product classifier model
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Initialize the pipeline for classifying product
product_classifier = pipeline("text-classification", model="Mahesh9/distil-bert-fintuned-product-cfpb-complaints",
                              max_length = 512, truncation = True, device = device)

# Load sub-product classifier models
with open('../subproduct_prediction/models/Credit_Reporting_model.pkl', 'rb') as f:
   trained_model_cr= pickle.load(f)
with open('../subproduct_prediction/models/Credit_Prepaid_Card_model.pkl', 'rb') as f:
   trained_model_cp= pickle.load(f)
with open('../subproduct_prediction/models/Checking_saving_model.pkl', 'rb') as f:
    trained_model_cs=pickle.load(f)
with open('../subproduct_prediction/models/loan_model.pkl', 'rb') as f:
   trained_model_l= pickle.load(f)
with open('../subproduct_prediction/models/Debt_model.pkl', 'rb') as f:
   trained_model_d= pickle.load(f)

@st.cache_resource(show_spinner=False)
# Define a function to select the appropriate subproduct prediction model based on the predicted product
def select_subproduct_model(predicted_product):
    if predicted_product == 'Credit Reporting' :
        return trained_model_cr
    elif predicted_product == 'Credit/Prepaid Card':
        return trained_model_cp
    elif predicted_product == 'Checking or savings account':
        return trained_model_cs
    elif predicted_product == 'Loans / Mortgage':
        return trained_model_l
    elif predicted_product == 'Debt collection':
        return trained_model_d
    else:
        raise ValueError("Invalid predicted product category")

# Loading the issue classifier model
issue_classifier = pipeline("text-classification", model="Mahesh9/distil-bert-fintuned-issues-cfpb-complaints",
                            max_length = 512, truncation = True, device = device)

# Path to the models and their corresponding names
issue_model_files = {
    'trained_model_account_operations': '../subproduct_prediction/issue_models/account_operations_and_unauthorized_transaction_issues.pkl',
    'trained_model_collect_debt': '../subproduct_prediction/issue_models/attempts_to_collect_debt_not_owed.pkl',
    'trained_model_closing_account': '../subproduct_prediction/issue_models/closing_an_account.pkl',
    'trained_model_closing_your_account': '../subproduct_prediction/issue_models/closing_your_account.pkl',
    'trained_model_credit_report': '../subproduct_prediction/issue_models/credit_report_and_monitoring_issues.pkl',
    'trained_model_lender': '../subproduct_prediction/issue_models/dealing_with_your_lender_or_servicer.pkl',
    'trained_model_disputes': '../subproduct_prediction/issue_models/disputes_and_misrepresentations.pkl',
    'trained_model_improper_use_report': '../subproduct_prediction/issue_models/improper_use_of_your_report.pkl',
    'trained_model_incorrect_info': '../subproduct_prediction/issue_models/incorrect_information_on_your_report.pkl',
    'trained_model_legal_and_threat': '../subproduct_prediction/issue_models/legal_and_threat_actions.pkl',
    'trained_model_managing_account': '../subproduct_prediction/issue_models/managing_an_account.pkl',
    'trained_model_payment_funds': '../subproduct_prediction/issue_models/payment_and_funds_management.pkl',
    'trained_model_investigation_wrt_issue': '../subproduct_prediction/issue_models/problem_with_a_company\'s_investigation_into_an_existing_issue.pkl',
    'trained_model_investigation_wrt_problem': '../subproduct_prediction/issue_models/problem_with_a_company\'s_investigation_into_an_existing_problem.pkl',
    'trained_model_credit_investigation_wrt_problem': '../subproduct_prediction/issue_models/problem_with_a_credit_reporting_company\'s_investigation_into_an_existing_problem.pkl',
    'trained_model_purchase_shown': '../subproduct_prediction/issue_models/problem_with_a_purchase_shown_on_your_statement.pkl',
    'trained_model_notification_about_debt': '../subproduct_prediction/issue_models/written_notification_about_debt.pkl',
}

issue_models = {}

for model_name, file_path in issue_model_files.items():
    with open(file_path, 'rb') as f:
        issue_models[model_name] = pickle.load(f)

# Define a function to select the appropriate subissue prediction model based on the predicted issue
def select_subissue_model(predicted_issue):
    if predicted_issue == "Problem with a company's investigation into an existing problem":
        return issue_models['trained_model_investigation_wrt_problem']
        
    elif predicted_issue == "Problem with a credit reporting company's investigation into an existing problem":
        return issue_models['trained_model_credit_investigation_wrt_problem']

    elif predicted_issue == "Problem with a company's investigation into an existing issue":
        return issue_models['trained_model_investigation_wrt_issue']

    elif predicted_issue == "Problem with a purchase shown on your statement":
        return issue_models['trained_model_purchase_shown']

    elif predicted_issue == "Incorrect information on your report":
        return issue_models['trained_model_incorrect_info']
        
    elif predicted_issue == "Improper use of your report":
        return issue_models['trained_model_improper_use_report']

    elif predicted_issue == "Account Operations and Unauthorized Transaction Issues":
        return issue_models['trained_model_account_operations']
        
    elif predicted_issue == "Payment and Funds Management":
        return issue_models['trained_model_payment_funds']

    elif predicted_issue == "Managing an account":
        return issue_models['trained_model_managing_account']
        
    elif predicted_issue == "Attempts to collect debt not owed":
        return issue_models['trained_model_collect_debt']

    elif predicted_issue == "Written notification about debt":
        return issue_models['trained_model_notification_about_debt']
        
    elif predicted_issue == "Dealing with your lender or servicer":
        return issue_models['trained_model_lender']

    elif predicted_issue == "Disputes and Misrepresentations":
        return issue_models['trained_model_disputes']
        
    elif predicted_issue == "Closing your account":
        return issue_models['trained_model_closing_your_account']

    elif predicted_issue == "Closing an account":
        return issue_models['trained_model_closing_account']
        
    elif predicted_issue == "Credit Report and Monitoring Issues":
        return issue_models['trained_model_credit_report']

    elif predicted_issue == "Legal and Threat Actions":
        return issue_models['trained_model_legal_and_threat']
        
    else:
        raise ValueError("Invalid predicted issue category")

# Custom Headers for enhancing UI Text elements
def custom_header(text, level=1):
    if level == 1:
        icon_url = "https://cfpb.github.io/design-system/images/uploads/logo_vertical_071720.png"
        # Adjust the img style as needed (e.g., height, vertical alignment, margin)
        st.markdown(f"""
            <h1 style="text-align: center;">
                <img src="{icon_url}" alt="Icon" style="vertical-align: middle; height: 112px; margin-right: -160px;">
                <span style="color: #008000; font-family: 'Sans Serif';">{text}</span>
            </h1>
        """, unsafe_allow_html=True)
        #st.markdown(f"<h1 style='text-align: center; color: #ef8236; font-family: sans serif;'>{text}</h1>", unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f"<h2 style='text-align: center; color: #00749C; font-family: sans serif;'>{text}</h2>", unsafe_allow_html=True)
    elif level == 3:
        st.markdown(f"<h3 style='text-align: center; color: #00749C; font-family: sans serif;'>{text}</h3>", unsafe_allow_html=True)
    elif level == 4:
        st.markdown(f"<h5 style='text-align: center; color: #00749C; font-family: sans serif;'>{text}</h5>", unsafe_allow_html=True)
    elif level == 5:
        st.markdown(f"<h5 style='text-align: center; color: #f63366; font-family: sans serif;'>{text}</h5>", unsafe_allow_html=True)

# Helper function for classifying the complaint
def classify_complaint(narrative):
    # Predict product category
    predicted_product = product_classifier(narrative)[0]['label']
    
    # Load the appropriate subproduct prediction model
    subproduct_model = select_subproduct_model(predicted_product)
    # Predict subproduct category using the selected model
    predicted_subproduct = subproduct_model.predict([narrative])[0]

    # Predict the appropriate issue category using the narrative
    predicted_issue = issue_classifier(narrative)[0]['label']
    
    # Load the appropriate subissue prediction model
    subissue_model = select_subissue_model(predicted_issue)
    # Predict subissue category using the selected model
    predicted_subissue = subissue_model.predict([narrative])[0]
    
    return {
        "Product" : predicted_product,
        "Sub-product" : predicted_subproduct,
        "Issue" : predicted_issue,
        "Sub-issue" : predicted_subissue
    }

# Helper function to display key insights
def plot_eda_charts(level):
    if level == 1:
        fig = complaints_by_year(df)
        return fig 
    
    if level == 2:
        fig = complaints_across_states(df)
        return fig 
    
    if level == 3:
        fig = plot_top_5_products(df)
        return fig

    if level == 4:
        fig = plot_top_5_issues(df)
        return fig
    
    if level == 5:
        fig = plot_top_5_issues_in_product(df)
        return fig
    
    if level == 6:
        fig = plot_top_10_companies_complaints(df)
        return fig
    
    if level == 7:
        fig = plot_top_10_states_most_complaints(df)
        return fig
    
    if level == 8:
        fig = plot_top_10_states_least_complaints(df)
        return fig
    
# Navigation setup
with st.sidebar:
    selected = option_menu(menu_title = "Navigate",
                       options = ["Home", "Key Insights", "Complaint Classifier"]
                       ,default_index = 0)
    
# Home Page
if selected == "Home":
    custom_header('CFPB Consumer Complaint Insights', level=1)
    # Introduction
    st.markdown("""
    <div style='text-align: center; color: #333; font-size: 20px;'>
        <p><strong>Uncover Consumer Trends and Automate Complaint Categorization with CFPB Insights</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.write("\n")

    # Project Motivation
    st.markdown("""
    ### :orange[Motivation]
    Consumers can face challenges with financial products and services, leading to complaints that may not always be resolved directly with financial institutions. The **Consumer Financial Protection Bureau (CFPB)** acts as a mediator in these scenarios. However, consumers often struggle to categorize their complaints accurately, leading to inefficiencies in the resolution process. Our project aims to **facilitate faster resolution** by automatically categorizing complaints based on narrative descriptions, enhancing the efficiency of complaint management.
    """, unsafe_allow_html=True)

    # Impact
    st.markdown("""
    ### :green[Impact]
    The implementation of our project has two primary impacts:
    - **Ease for Consumers:** Automates the tagging of complaints into appropriate categories, reducing the need for consumers to understand complex financial product categories.
    - **Industry Adoption:** Offers a streamlined approach to complaint handling that can be adopted by financial institutions beyond the CFPB, promoting consistency across the industry.
    """, unsafe_allow_html=True)
    # Complaint Classifier
    st.markdown("""
    #### :blue[Complaint Classifier]
    Our dashboard features an innovative :rainbow[**Complaint Classifier**] that utilizes the narrative descriptions provided by consumers to categorize complaints into the correct product, issue, and sub-issue categories. This tool simplifies the submission process for consumers and enhances the efficiency of complaint resolution.
    """, unsafe_allow_html=True)

# Key Insights Page
elif selected == "Key Insights":

    headers = ["Evolution of complaints across years", "Complaints across US states",
               "Top 5 Common Product Categories", "Top 5 Common Issue Categories",
               "Top 5 Issues in Each Product Category", "Top 10 Companies with Most Complaints in 2023",
               "Top 10 states with Most Complaints", "Top 10 states with Least Complaints"]

    custom_header("Key Insights", level=1)
    st.write("\n")
    st.write("\n")
    st.write("\n")

    for i in range(0, len(headers), 2):
        cols = st.columns(2)  # Create two columns
        
        with cols[0]:
            custom_header(headers[i], level=4)
            fig = plot_eda_charts(level=i+1)
            st.plotly_chart(fig, use_container_width=True)
        
        if (i+1) < len(headers):
            with cols[1]:
                custom_header(headers[i+1], level=4)
                fig = plot_eda_charts(level=i+2)
                st.plotly_chart(fig, use_container_width=True)

# Complaints Classifier Page
elif selected == "Complaint Classifier":
    custom_header("Complaint Classifier", level=2)
    st.write("\n")
    
    # Using a key for the text_area widget to reference its current value
    query = st.text_area("Enter your complaint:", placeholder="It is absurd that I have consistently made timely payments for this account and have never been overdue. I kindly request that you promptly update my account to reflect this accurately.", key="input_text")
    if st.button("Classify Complaint"):
        with st.spinner("Classifying Complaint..."):
            result = classify_complaint(query)
            if result:  # Check if the result is not empty
                st.success("Complaint Classification Results:")
                
                #Using HTML for better control over formatting
                st.markdown(f"""
                **Product:** :blue[{result.get("Product")}]<br>

                **Sub-product:** :green[{result.get("Sub-product")}]<br>

                **Issue:** :red[{result.get("Issue")}]<br>

                **Sub-issue:** :orange[{result.get("Sub-issue")}]<br>

                """, unsafe_allow_html=True)
                st.write("\n\n")
                st.header("", divider= 'rainbow')
            else:
                st.error("Failed to classify the complaint. Please try again.")
            #time.sleep(1)
            st.balloons()  # Celebratory balloons on successful classification