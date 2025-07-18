import nltk

# Download into a folder that Render will cache between builds
nltk.download('stopwords', download_dir='nltk_data')
nltk.download('wordnet',   download_dir='nltk_data')
