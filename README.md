# News Source Identifier 

Data comes from this collection: https://www.kaggle.com/snapcrack/all-the-news/home

## TODO
- [x] What data? - gatherd from Kaggle
- [x] How is it cleaned/prepared? - names are scrubbed from doc body and then
  tokenized
- [ ] What metrics are we using? 
- [ ] What is the baseline?
- [ ] How does baseline perform? 


## RUN
- Save article csvs from kaggle into `data/`

- To run the sentiment model, you need to uncomment some lines in `sentiment.py`
  where you save the data. After saving the data, you can run `python3
  sentiment_model.py` to run the sentiment model and get the confusion matrix.

- To run the ML + Sentiment model, cd into `experiments` and run `python3
  data_reader.py ../data/articles1.csv ...` passing all articles csvs are
  command line arguments. This will take a while but it will generate the data
  and the confusion matrix. 
