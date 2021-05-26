# News headers classification
Project created for one of the recruitment process

Language:
 * English (Notebook)


## About 
The main objective of this project was to build a classifier for news headers. It was my first time with multiclass classification problem, I have worked with binary problems so far. The data contained news headers (text), several other columns (which were deleted) and target variable (4 classes - business, entertainment, health amd technology and science). The biggest part of this project was text data analysis, cleaning and feature engineering (NLP). Target variable was analyzed (number of observations and balance check) and the data were splitted into train, validation and test set. Analysis and feature engineering was based on train set only and applied to all the sets. First, random training observations were inspected to get more familiar with the problem, next some cleaning with regex was done. Each change in the data structure was verified with Word Clouds independent for each class. The next step was to analyze standard clouds and build stopword list based on NLTK and WORDCLOUD packages extended by own ideas. Next step was to perform tokenization and lemmatization. Two lemmatizers was considered - WordNet and SpaCy. The secend one works extremely well and was used. After that preprocessing steps feature enigieering and the whole fun began. The idea was to build a two input neural network with two separated branches concatenated at the end and processed to the softmax classifier. First branch was a Convolutional one, so the GloVe word embeddings were used. The second one was dense branch which was baseed on engineered features. The new features was based on the headers length (original header length, preprocessed header length and mean number of words per header, all normalized with standard (x-mu)/sigma formula). Next some potential were inspected based on separated histograms and density plots for each class and only the good features were kept. The next feature was header sentiment computed with VADER sentiment alayzer (and normalized with the same formula as length features). It turned out that science and technology category has the most neutral sentiment (based on density plots). Next features was based on the SpaCy Named Entity Recognition and number of tags groups (Persons, organizations, places, etc.) The tags were analyzed a bit deeper and the most frequent ones (for example Kim Kardashian, Apple, Fed, Samsung) were used as separated features. The last part of feature engineering for dense brach was n-grams analysis. Several most frequent biggrams and unigrams were used (ebola outbreak, wall street, game throne, cancer, stock or iphone) as another features. Like it was mentioned before, another branch was built based on convolutional layers so the feature engineering for this part was based on pretrained GloVe word embedding vectors (100-dimensional). Some coverage analysis was performed to see how many words are covered with GloVe vectors. It turned out that detailed cleaning paid off and the coverage was around 98.5% and 81-89% for uniqie words (train and validation set). Of course some outliers was spotted - for example 0 length titles (all the words were from stoplist) or some bad examples (title combined with URL 100>words). Outliers were deleted. What is more the idea was to build CNN with filtersize = 3, so the headers shorter than 3 were deleted too (near 10 000 obsrevations, but the dataset was pretty big - near 400 000 observations).

The next part of this project was about model building and validation, and the final model was built with embeddings + 1D CNN (filter size 3 and relu activation) and 1D Max Poolig (2 filter size) all that was flattened and concatenated with dense branch output. Dense branch was built with only one dense layer wit 60 units and relu activation (65 features was engineered in total). After concatenation, the main branch size was 830 units what was densely connected with next 430 units with relu activation, 0.0005 L2 regularization and 0.3 dropout. The next layer was dense with 215 units, relu and 0.3 dropout. All that connected with softmax classifier with 4 outputs. Categorical Crossentropy loss function and Adam optimizer with default settings.

The model was trained in 256 batches for 20 epochs and the results were very high 91% accuracy on medical, technology (and science) and business and 97% for entertainment (validation set). Test results confusion matrix is presented below:

![](https://raw.githubusercontent.com/maciejodziemczyk/news-headers-classification/main/confusion_matrix.png)

As you can see, the test results was very high and overfitting wasn't a problem.

Findings:
 - Feature engineering is extremely important when working with text data
 - neural networks are extremely good and flexible, two input model can take word embeddings advantage and also use information lost by them using another dense branch built on engineered features
 - the data were collected at the ebola outbreak time and Miley Cyrus was very popular then. Kim Kardashian, Kanye West and Jay Z was very popular too. Samsung, Microsoft, Google, Apple and Facebooks are company names mentioned very often in news headers. Business was around USA, China, Russia and Iraq.
 - Glove embeddings can be used for this type of task paired with 1D CNN and 1D max pooling.

In this project I learnt a lot about Natural Language Processing, especially cleaning and feature engineering. I gained more experience with neural networks amd Python too. I really enjoy this type of projects. NLP is fascinating.

## Repository description
 - News_classification.ipynb - Jupyter Notebook with whole analysis (Python)
 - the data was stroed in data catalog and downloaded from [ML UCI repository](http://archive.ics.uci.edu/ml/datasets/News+Aggregator)
 - I didn't upload created data folders because of its size (you can generate it with notebook)
 - GloVe embeddings was stored in GloVe catalog and can be downloaded from [Stanford page](https://nlp.stanford.edu/projects/glove/)

## Technologies
 - Python (numpy, pandas, matplotlib, seaborn, nltk, spacy, regex, scikit-learn, keras)
 - Jupyter Notebook

## Author
 - Maciej Odziemczyk
