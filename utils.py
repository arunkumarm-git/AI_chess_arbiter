import streamlit as st


def firstmessage():
    st.toast("Hooray! You're a chess viber now", icon="🎉")
    st.balloons()


def other_tools():
    for i in range(6):  # For space
        st.write(" ")

    st.link_button("Vibe with chess", "https://vibewithchess.com")
    st.link_button("Chess review", "https://chessvibe.onrender.com/")
