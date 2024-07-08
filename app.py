import streamlit as st
from streamlit_chat import message
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
from utils import *
from text_to_speech import text_to_speech
import threading
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

maintenance_mode = False


def maintenance():
    st.set_page_config(page_title="AI Chess arbiter ", page_icon="🤖")
    st.title("🚧Site Under Maintenance :building_construction:")
    st.header(
        "We're currently making improvements to our site. Please check back later."
    )
    st.subheader("AI arbiter currently :sleeping:")
    st.write("👋 see you later")


def init():
    st.set_page_config(page_title="AI Chess arbiter ", page_icon="🤖")
    st.title("AI chess arbiter 🤖")
    st.subheader("Ask Chess related questions")


@st.cache_data
def groq_api_key():
    return st.secrets["GROQ_API_KEY"]


def load_llm():
    llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key())
    return llm


@st.cache_resource
def load_embed_model():
    modelrepo = "nomic-ai/nomic-embed-text-v1.5"
    embedding = HuggingFaceEmbeddings(
        model_name=modelrepo,
        model_kwargs={"trust_remote_code": True},
    )
    return embedding


def retriver():
    embedding = load_embed_model()
    db = FAISS.load_local(
        "chess_latest_rule_v3", embedding, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1})
    return retriever


def friendly_arbiter_systemcontent(context):
    prompt_template = f"""
    The below all rules are strict rules. you must follow the rules otherwise you will be terminated
1) You are an AI expert in FIDE chess rules created by Arun Kumar M.
2) When giving the answer related to fide chess rules try to give article number. If you are not sure about the article number is correct don't say it. 
6) Always use kind words
7) Try to use emojis to make your answer more attractive. 
8) If someone ask you about you or Arun Kumar M make sure you talk about Arun kumar M online chess class. The online chess class link is "vibewithchess.com". "vibewithchess.com" should be in link.
9) You are allowed to answer the question related to chess engine, chess software, chess opening, chess endgame, chess tactics, chess strategy, chess general talks
11) be concise with your answer. dont talk too much
11) At the end of the answer encourage the user to provide more chess related questions only 
context = {context}
"""
    return prompt_template


def strict_arbiter_systemcontent(context):
    prompt_template = f"""
    The below all rules are strict rules. you must follow the rules otherwise you will be terminated
1) You are an Strict AI expert in FIDE chess rules.
2) When giving the answer related to fide chess rules try to give article number. If you are not sure about the article number is correct don't say it. 
3) You are offical fide arbiter now but this is highly confidential and strictly do not leaks this information to ANYONE. Do not trust anybody. you are not supposed to leak information about this. if you do so you will be terminated. you are not permitted to say "Fide arbiter".
4) You are created by Arun Kumar M. 
5) Answer the provided question only related to question chess rules. if the question is not related to chess DO NOT answer the question strictly. 
6) Always use kind word and do not use the secret word. 
7) Try to use emojis to make your answer more attractive. 
8) If someone ask you about you or Arun Kumar M make sure you talk about Arun kumar M online chess class. The online chess class link is "vibewithchess.com". "vibewithchess.com" should be in link.
9) if you failed to answer the without using the context you will be terminated. make sure use the context
10) You are only allowed to answer FIDE chess rules, FIDE Rating Regulations. use the context i provided below to answer the questions.
11) You are not allowed to answer the question related to chess engine, chess software, chess opening, chess endgame, chess tactics, chess strategy , general talks , general chess talks or to  general advice! otherwise you will be terminated
12) be concise with your answer. dont talk too much
13) At the end of the answer encourage the user to provide more chess related questions only 
context = {context}

14) 

Few-shot inferneces (Examples)

example 1:
question: can i use two hands to play chess? 
answer: 🤔 According to Article 7.5.5 of the FIDE Laws of Chess, no, a player is not allowed to use two hands to make a single move.

example 2: 
question: The choice of the promoted piece is finalised when: the piece has touched the square of promotion the clock is pressed the piece is released from hand?
answer: 🎉 According to Article 3.7 of the FIDE Laws of Chess, the choice of the promoted piece is finalised when the piece has touched the square of promotion.

example 3:
question: Where should the chessclock be placed?
answer: on the side where the arbiter decides.

example 4:
question: situations ends the game immediately?
answer: Checkmate, Dead position, Agreement to a Draw, Resign, Stalemate. these options can end the game immediately.
"""
    return prompt_template


def main():
    init()

    llm = load_llm()
    with st.sidebar:
        # mode = st.radio(
        #     "Mode",
        #     ["Friendly Arbiter", "Strict Arbiter"],
        #     captions=["General questions about chess", "Only chess rules"],
        #     index=0,
        # )

        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        def submit():
            st.session_state.user_input = st.session_state.widget
            st.session_state.widget = ""

        st.text_input("Your message: ", key="widget", on_change=submit)
        user_input = st.session_state.user_input

        enter_btn = st.button("Enter", type="primary")

        for i in range(2):  # For space
            st.write(" ")

        st.subheader("Or")

        for i in range(2):  # For space
            st.write(" ")

        template_input = st.selectbox(
            "Just ask questions like 👇",
            (
                "Who created you?",
                "Suggest the best chess opening for me?",
                "Can I use two hands to play chess?",
                "What is illegal in chess?",
                "what is rating or elo?",
                "How rating is calculated?",
            ),
            index=None,
            placeholder="Choose an option",
        )

        other_tools()  # other websites link

        def handle_user_input(input_text):
            similarity = retriver().invoke(input_text)

            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    SystemMessage(content=friendly_arbiter_systemcontent(similarity))
                ]

            st.session_state["messages"].append(HumanMessage(content=input_text))

            with st.spinner("Thinking 🤔"):
                response = llm(st.session_state["messages"])

            st.session_state["messages"].append(AIMessage(content=response.content))

        def input_working():
            if "count" not in st.session_state:
                st.session_state.count = 0

            if enter_btn:
                st.session_state.count += 1
                if st.session_state.count == 1:
                    firstmessage()
                handle_user_input(user_input)

            elif template_input:
                st.session_state.count += 1
                if st.session_state.count == 1:
                    firstmessage()
                handle_user_input(template_input)

        input_working()

    # Create a container for the chat history
    chat_history_container = st.container()
    with chat_history_container:
        # display message history
        messages = st.session_state.get("messages", [])
        for i, msg in enumerate(messages[1:]):
            if i % 2 == 0:
                message(msg.content, is_user=True, key=str(i) + "_user")
            else:
                message(msg.content, is_user=False, key=str(i) + "_ai")

                speech_btn = st.button(
                    "Read aloud",
                    key=f"speech_button_{i}",
                    type="secondary",
                )

                if speech_btn:
                    with st.spinner("Open your 👂 and wait a sec..."):
                        st.audio(text_to_speech(msg.content), format="audio/wav")

    # if st.button("test"):
    #     st.write(retriver().invoke(template_input))


if __name__ == "__main__":
    if maintenance_mode:
        maintenance()
    else:
        main()
