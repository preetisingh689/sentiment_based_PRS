from flask import Flask, request, render_template
from model import SentimentAnalysis
from model import RecommendationSystem
import pickle
import pandas as pd

# create an instance
app = Flask(__name__)

df = pd.read_csv("sample30.csv")
saverecomfile = RecommendationSystem(df)
user_final_rating = pd.read_csv("user_final_rating.csv")
user_final_rating = user_final_rating.set_index('reviews_username')


@app.route("/", methods=['POST', "GET"])
def home():
    if request.method == 'POST':
        user = request.form['name']
        if user in user_final_rating.index.tolist():
            lis = user_final_rating.loc[user].sort_values(ascending=False).index[:20]
            df_recom = df[df['name'].isin(lis)]
            final_df = SentimentAnalysis(df_recom)
            final_df = final_df[['name', 'prediction']]
            final_df['pred_num'] = final_df.prediction.replace({'Positive':1,'Negative':0})
            d = final_df.groupby('name').mean().sort_values(ascending=False, by="pred_num")*100
            products = d[:5].index.tolist()
            return render_template('index.html', products=products, submit="yes")
        else:
            return render_template('index.html', products="None")
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
