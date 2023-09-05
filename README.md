# Sentiment Analysis of Tokopedia App User

## Project Overview
Every app has users who can give their opinions about the app. Their opinion can also contribute to the app's improvement in the future so it is important to know their opinion either negative or positive. The negative opinions and positive opinions have different treatment so we have to classify where the negative comments and positive comments are. But we can't handle a lot of user reviews by classifying them ourselves, so we need a machine to do this job. 
- Classify whether an app review is negative or positive.
- The data was taken by scraping user reviews of the Tokopedia app on the Google PlayStore. Only collected Indonesian language reviews for a year and only used 6,000 resample data.
- Deploy a Machine Learning model using Flask so the end-users can input a review there and get the result of classification directly. [Here](https://drive.google.com/file/d/1wvcGmQlugdEotN61mwyp0OCQOcwQ1BYJ/view?usp=sharing) is the screen recording of the web app on my local page.

## Objectives
* The objective is to make sentiment classification based on the review. As a result, the model can help company to know what the negative opinions about their app so they can improve them and the positive opinions so they can maintain that feature or service.

## Methodology
- **Scraping Data**

  Scraped data from Google Playstore using `google-play-scraper` package.

- **Preparing Data**

  Determined the sentiment variable which would have a "negative" value or (0) if the score or rating value was ≤ 3 and have a "positive" value or (1) if the score or rating value was > 3. After that, resampled data without replacement with only 3,000 data for each sentiment so data would be balance.
  
- **Preprocessing Data**

  The reviews data in the form of text would be processed first by doing a case folding, stopword removal (used a combination of stopwords from the Sastrawi and NLTK Indonesian language libraries as well as the document/corpus stopwords [here](https://github.com/wandalistathea/analisis_sentimen_tokopedia/blob/main/list%20stopword%20baru%20(tambahan%20sendiri).txt) made by self by taking less important words from the data where these words have not been deleted by the two libraries), writing correction (for words that were not in the standard form or there were typos like [this](https://github.com/wandalistathea/analisis_sentimen_tokopedia/blob/main/list%20koreksi%20penulisan%20(tambahan%20sendiri).txt)), and also stemming (used Sastrawi library).

- **Exploratory Data Analysis**

  Visualized using word cloud for each sentiment.
  
- **Weighting (Value on Text Data)**

  Used TF-IDF (Term Frequency-Inverse Document Frequency).
  
- **Modeling and Model Evaluation**

  The method used was Support Vector Machine (SVM) using the accuracy value of 10-fold cross validation as an evaluation model.

## Web App (Flask)
Code for deployment can be accessed in [this folder](https://github.com/wandalistathea/analisis_sentimen_tokopedia/tree/main/Deployment). Use this command below in the terminal to automatically install all the packages needed and run the web app

    pip install -r requirements.txt

    python app.py

## Conclusions
By using the Support Vector Machine (SVM) classification model, the accuracy value of the 10-fold cross validation is **87.19%**.
