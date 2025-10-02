# Phase 1 of the FIAM Hackathon

## Goal
Our goal is to predict the stock return for next month based on previous financial data of several company.
From there, we will select the top companies to go *long* (buy), and *short* (sell ?) the ones with lowest predicted return.

## Strategy
- 1 : Preprocess financial data, train a AI model to predict next month's stock return
- 2 : Use a pre-trained LLM to analyze sentiments from financial report
- 3 : Use a web scrapping and a pre-trained LLM to gather real-time information and predict impact on stocks
- 4 : Win first prize and gamble with it (optional)


## Installation
On the main folder, run 
``` pip install -r requirements.txt ``` from the cmd

Make sure to put ```ret_sample.csv``` in the main folder too.

## Usage
The ```LSTM.py``` file contains two AI models for stage one. Only use LSTM_seq2one tho, the other is shit.

Run ```unpack_data.py``` to get a cleaner and usable version of ```ret_sample.csv``` ( it may take a while to run, and takes a lot of space)

Run ```LSTM.py```to train an AI on some company data.


```run.py``` gives an example of the model's output, and compares it to real expected stock return value.

```finBert.py```is a basic implementation of the finBERT LLM. It takes a *string* as an input and output a *dict['sentiment','score']*

### Counter

Caffeine: 990 mg

Time spent: 25 Hours

Nb sessions: 6


Please don't plagiarise my work for a hackathon, or at least make sure to mention my project. Plagiarism is bad, and can be severly sanctioned. You can however contact me at hehugo1024@gmail.com if you want to discuss about the project, or if you need any guidance.