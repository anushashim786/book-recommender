import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(page_title="Book Recommender System", page_icon="📚")

st.title("📚 Book Recommender System")
st.write("Select a book you like, and our Machine Learning model will recommend similar books for you!")

# 1. Books Dataset
books_data = {
    'Title': [
        'Atomic Habits', 'The Power of Habit', 'Deep Work', 
        'Rich Dad Poor Dad', 'The Psychology of Money', 'Think and Grow Rich',
        'Harry Potter and the Sorcerer\'s Stone', 'The Hobbit', 'Percy Jackson: The Lightning Thief',
        'Sherlock Holmes: A Study in Scarlet', 'The Da Vinci Code', 'Gone Girl',
        'The Alchemist', 'The Kite Runner', 'Forty Rules of Love'
    ],
    'Author': [
        'James Clear', 'Charles Duhigg', 'Cal Newport',
        'Robert Kiyosaki', 'Morgan Housel', 'Napoleon Hill',
        'J.K. Rowling', 'J.R.R. Tolkien', 'Rick Riordan',
        'Arthur Conan Doyle', 'Dan Brown', 'Gillian Flynn',
        'Paulo Coelho', 'Khaled Hosseini', 'Elif Shafak'
    ],
    'Genre_Keywords': [
        'self-help habits productivity personal growth success',
        'self-help habits psychology behavior routine',
        'productivity focus work habits self-help concentration',
        'finance money investing wealth business personal finance',
        'finance money psychology wealth investing behavior',
        'finance success wealth mindset business self-help',
        'fantasy magic adventure wizards fiction young adult',
        'fantasy adventure magic dragons fiction middle earth',
        'fantasy adventure mythology magic fiction young adult',
        'mystery detective crime thriller suspense investigation',
        'mystery thriller suspense history secrets puzzle',
        'thriller mystery suspense psychological crime drama',
        'fiction philosophy spiritual journey destiny inspiration',
        'fiction drama emotion friendship historical redemption',
        'fiction spiritual love philosophy historical drama'
    ]
}

df = pd.DataFrame(books_data)

# 2. Machine Learning: TF-IDF Vectorization & Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Genre_Keywords'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Recommendation Function
def recommend_books(selected_book):
    idx = df[df['Title'] == selected_book].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_books_indices = [i[0] for i in sim_scores[1:4]]
    return df.iloc[top_books_indices]

# 4. Streamlit User Interface
selected_book = st.selectbox("Choose a book you enjoyed:", df['Title'].values)

if st.button("Recommend Similar Books 🔍"):
    st.subheader(f"Because you liked '{selected_book}', you might also enjoy:")
    
    recommendations = recommend_books(selected_book)
    
    for index, row in recommendations.iterrows():
        st.success(f"📖 **{row['Title']}** — *by {row['Author']}*")

# Option to view dataset
with st.expander("View Complete Books Dataset"):
    st.dataframe(df)
