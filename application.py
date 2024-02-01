import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
import sklearn

ps = PorterStemmer()


def text_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def main():

    st.set_page_config(page_title="Spam Classifier", layout="wide")

    # Use local CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")

    # ---- HEADER SECTION ----
    with st.container():
        st.title("Streamlined SMS Spam Detection")
        st.subheader("Effortlessly Identify Spam or Legitimate Messages")
        st.write(
            "Utilize our SMS Spam Classifier by simply inputting your text into the provided text box. "
            "Click the Classify button, and within moments, our advanced algorithm analyzes the content, "
            "distinguishing between spam and legitimate messages. "
            "The result is promptly displayed, indicating whether the input text is classified as spam or not. "
            "Streamlining the process, our user-friendly interface ensures efficient spam detection with ease."
        )
        st.write("[Learn More about this project >](https://pythonandvba.com)")

    # ---- CLASSIFIER ----
    with st.container():
        st.write("---")
        left_column, middle_column, right_column = st.columns([5, 2, 3])

        with left_column:
            st.subheader("Enter text below: ")
            st.write("##")
            input_sms = st.text_area(" ")
            if st.button('Predict'):
                transformed_sms = text_preprocess(input_sms)
                vectored_input = tfidf.transform([transformed_sms])
                result = model.predict(vectored_input)[0]
                if result == 1:
                    st.header("Spam")
                else:
                    st.header("Not Spam")
        img = "text-messaging.jpg"

        with middle_column:
            st.write("##")

        with right_column:
            st.image(img, width=350)

    # ---- CONTACT ----
    with st.container():
        st.write("---")
        st.subheader("Get In Touch With Me!")
        st.write("##")

        contact_form = """
        <form action="https://formsubmit.co/aparajitanegisingh@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()

if __name__ == "__main__":
    main()
