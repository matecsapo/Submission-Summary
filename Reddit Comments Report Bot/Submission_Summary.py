#Set up Reddit API Access
import praw
reddit = praw.Reddit(
    client_id = "zfUybHJUxmGCAbgmhFzxXw",
    client_secret = "6TG5GlpR6alXFW5cMJu4BaMzde_PdA",
    user_agent = "Submission Summary by Football_Forecast",
    username = "Football_Forecast",
    password = "leledoba22"
)

#Import BERT Extraction Summarizer
from summarizer import Summarizer
Extractive_Summarizer = Summarizer()

#Import BART Abstractive Summarizer
from transformers import BartTokenizer, BartForConditionalGeneration
# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

#Import NLTK (natural langauge processing toolkit)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
Sentiment_Analyzer = SentimentIntensityAnalyzer()
#Specialized extra weightings
Caps_Lock_Weight = 0.50
Z_W_Weight = 0.75

#Import python time module
import time

#Submission to analyze
#Submission_ID = "1g5t2ar"

#Useful websites
#https://medium.com/@sarowar.saurav10/6-useful-text-summarization-algorithm-in-python-dfc8a9d33074
#https://medium.com/@sandyeep70/demystifying-text-summarization-with-deep-learning-ce08d99eda97
#https://medium.com/@tulasids/streamlining-text-summarization-with-hugging-faces-bart-model-8f8ada8e8508

#SubReddits to analyze
Served_Subreddits = ("technology")

#Comment Extraction Recursive Helper
def Comment_Extraction_Helper(Root_Comment, CC_Upvote_Threshold):
    #Lists for storing comment forest stemming from this root
    Comments = []
    Critical_Comments = []
    #add root comment onto the list
    Comments.append(Root_Comment.body)
    if Root_Comment.score >= CC_Upvote_Threshold:
        Critical_Comments.append(Root_Comment.body)
    #if the comment has replies, add those on as well    
    if len(Root_Comment.replies) > 0:
        for child_comment in Root_Comment.replies:
            All_Comments, Only_Critical_Comments = Comment_Extraction_Helper(child_comment, CC_Upvote_Threshold)
            Comments.extend(All_Comments)
            Critical_Comments.extend(Only_Critical_Comments)
    #return comment list stemming from this root comment        
    return Comments, Critical_Comments

#Critical Comments Formatter
def Critical_Comments_Format(Comments):
    Combined_Comments = ""
    for comment in Comments:
        #remove endlines
        comment = comment.strip('\n')
        comment = comment.strip('\t')
        comment = comment.strip('\r')
        Combined_Comments += "." + comment + " "
    return Combined_Comments    

#BART Abstractive Summarization
def Abstractive_Summarizer(Input_Text):
    # Tokenize and summarize the input text using BART
    inputs = tokenizer.encode("summarize: " + Input_Text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=175, min_length=125, length_penalty=2.0, num_beams=4, early_stopping=True)
    #Generate Abstractive Summary
    Abstractive_Summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return Abstractive_Summary

#Sentiment Score Calculator
def Sentiment_Score(Comments):
    #initialize vars for storing data
    Average_Pos_Score = 0
    Average_Neg_Score = 0
    Average_Compound_Score = 0
    Zero_Words = 0
    #Iterate over all Comments
    for i in range (len(Comments)):
        #Prep the comments for analysis
        tokens = word_tokenize (Comments[i])
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        Processed_Word = ' '.join(lemmatized_tokens)
        #Obtain sentiment scores
        Average_Pos_Score += Sentiment_Analyzer.polarity_scores(Processed_Word)["pos"]
        Average_Neg_Score += Sentiment_Analyzer.polarity_scores(Processed_Word)["neg"]
        Average_Compound_Score += Sentiment_Analyzer.polarity_scores(Processed_Word)["compound"]
        #Consider the special weightings
        if Sentiment_Analyzer.polarity_scores(Processed_Word)["compound"] == 0 and Processed_Word.isupper():
            if Average_Compound_Score > 0:
                Average_Compound_Score += Caps_Lock_Weight
            elif Average_Compound_Score < 0:
                Average_Compound_Score -= Caps_Lock_Weight
        elif Sentiment_Analyzer.polarity_scores(Processed_Word)["compound"] == 0:
            Zero_Words += 1
    #Final average sentiment scores        
    Average_Compound_Score = Average_Compound_Score / (len(Comments) - Z_W_Weight*Zero_Words)
    Average_Pos_Score = Average_Pos_Score / (len(Comments) - Z_W_Weight*Zero_Words)
    Average_Neg_Score = Average_Neg_Score / (len(Comments) - Z_W_Weight*Zero_Words)
    return Average_Compound_Score

#Comment creator + poster
def Comment_Report(Post_ID):
    #Create an instance of the reddit post we are analyzing
    post = reddit.submission(id=Post_ID)
    print("Post Instance Successfully Created for --> " + post.title)

    #Extract the comments
    Comments = []
    Critical_Comments = [] #comments which have significant traction (> a certain # of upvotes)
    #Calculate critical_comments threshold
    CC_Upvote_Threshold = 0.022*post.score
    #Iterate over comment forest
    post.comments.replace_more(limit=None)
    for top_level_comment in post.comments:
        All_Comments, Only_Critical_Comments = Comment_Extraction_Helper(top_level_comment, CC_Upvote_Threshold)
        Comments.extend(All_Comments)
        Critical_Comments.extend(Only_Critical_Comments)
    print("Comments Extracted Successfully")        
    
    #Format Critical Comments
    Critical_Comments_Formatted = Critical_Comments_Format(Critical_Comments)
    print("Critical Comments Formatted Successfully")

    #Generate Extractive Summarization
    #Extractive_Summary = Extractive_Summarizer(Critical_Comments_Formatted, num_sentences = 3)
    #Extractive_Summary = Extractive_Summary.strip('\n')
    #Extractive_Summary = Extractive_Summary.strip('\t')
    #Extractive_Summary = Extractive_Summary.strip('\r')
    #print("Extractive Summary Generated Successfully")

    #Generate Abstractive Summarization
    Abstractive_Summary = Abstractive_Summarizer(Critical_Comments_Formatted)
    print("Abstractive Summarization Generated Successfully")

    #Calcualte average sentiment score values
    Compound_Sentiment_Score = Sentiment_Score(Comments)
    Compound_Sentiment_Score = round(Compound_Sentiment_Score, 4)
    print("Sentiment Score Calculated Successfully")

    #Post Summary report comment to the submission
    #post.reply("<--- Submission Summary --->\n\n*** Post Title --> " + post.title + "\n\n*** Abstractive Comments Summary --> " + Abstractive_Summary + "\n\n*** Extractive Comments Summary --> " + Extractive_Summary + "\n\n*** Collective Comments Positivity/Negativity Score --> " + str(Compound_Sentiment_Score) + "\n\n<--- Report created by Submission Summary Bot. Upvote if you found this useful so others see it too! --->")
    post.reply("<--- Submission Summary --->\n\n*** Post Title --> " + post.title + "\n\n*** Abstractive Comments Summary --> " + Abstractive_Summary + "\n\n*** Collective Comments Positivity/Negativity Score --> " + str(Compound_Sentiment_Score) + "\n\n<--- Report created by Submission Summary Bot. Upvote if you found this useful so others see it too! --->")
    print("Comment report succesfully generated for --> " + post.title + "\n\n")
    #print("\n\n<--- Submission Summary --->\n\n*** Post Title --> " + post.title + "\n\n*** Abstractive Comments Summary --> " + Abstractive_Summary + "\n\n*** Extractive Comments Summary --> " + Extractive_Summary + "\n\n*** Collective Comments Positivity/Negativity Score --> " + str(Compound_Sentiment_Score) + "\n\n<--- This report was created by the Submission Summary Bot. Upvote if you found this useful so others see it too! --->")


#Iterate over all comments in served subreddits
for post in reddit.subreddit(Served_Subreddits).hot(limit=30):
    #If Post has a sufficient # of comments, and is not a pinned post, produce a comment report of it
    if post.num_comments >= 120 and not post.stickied:
        #Run comment report on a specific submission
        Comment_Report(post.id)
    #Pause to get around reddit API ratelimit
    time.sleep(1)