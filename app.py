
import streamlit as st
from pytube import YouTube
import os
import whisper
from moviepy.editor import *
import datetime as dt
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


def load_video(url):
    yt = YouTube(url)
    #video_stream = yt.streams.filter(file_extension="mp4")
    target_dir = os.path.join('/content')

    #yt.streams.first().download()(output_path=target_dir)
    st.write('----DOWNLOADING VIDEO FILE----')
    file_path = yt.streams.filter(only_audio=True, subtype='webm', abr='160kbps').first().download(output_path=target_dir)
    return file_path



def process_video(path):
    file_dir = path
    st.write('Transcribing Video with whisper base model')
    model = whisper.load_model("base")
    result = model.transcribe(file_dir)
    return result



def process_text(result):
    texts, start_time_list = [], []
    for res in result['segments']:
        start = res['start']
        text = res['text']

        start_time = dt.datetime.fromtimestamp(start)
        start_time_formatted = start_time.strftime("%H:%M:%S")

        #creating list of texts and start_time
        texts.append(''.join(text))
        start_time_list.append(start_time_formatted)

    texts_with_timestamps = dict(zip(texts, start_time_list))

    formatted_texts = {
        text: dt.datetime.strptime(str(timestamp), '%H:%M:%S')
        for text, timestamp in texts_with_timestamps.items()
    }

    #grouping the sentences in the interval of 30 seconds, & stoding the texts and starting time
    # in group_texts & time_list reps

    grouped_texts = []
    current_group = ''
    time_list = [list(formatted_texts.values())[0]]
    previous_time = None
    time_difference = dt.timedelta(seconds=30)

    # Group texts based on time difference
    for text, timestamp in formatted_texts.items():

        if previous_time is None or timestamp - previous_time <= time_difference:
            current_group += text
        else:
            grouped_texts.append(current_group)
            time_list.append(timestamp)
            current_group = text
        previous_time = time_list[-1]

    # Append the last group of texts
    if current_group:
        grouped_texts.append(current_group)

    return grouped_texts, time_list



def get_vectorstore(grouped_texts, time_list):

    text = grouped_texts
    time_stamps = time_list

    time_stamps = [{'source': str(t.time())} for t in time_stamps]
    embeddings = HuggingFaceEmbeddings()
    vectorestore = FAISS.from_texts(texts=text, embedding=embeddings, metadatas=time_stamps)
    return vectorestore



def get_conversation(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain


def main():

    favicon_url = 'https://th.bing.com/th/id/R.087b4dc55ac459f86e6d11d402095394?rik=SfrwQLE7z60OLg&pid=ImgRaw&r=0&sres=1&sresct=1'
    st.set_page_config(page_title='Chat with YouTube videos', page_icon=favicon_url)
    st.header('Chat with your videos :film_frames:')
    user_question = st.text_input('Enter your query here')

    if 'vectorstore' not in st.session_state:
      st.session_state.vectorstore = None


    if 'message' not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


    with st.sidebar:
        st.subheader('Your Video file')
        url = st.text_input('Enter your URL here and click on "Process"')
        if st.button('Process'):
            with st.spinner('Processing'):

                st.video(url)

                #load the video
                path = load_video(url)

                #convert audio to text file using whisper
                result = process_video(path)
                st.write(result)

                #Embeed & transfer the converted text into vectorstore
                grouped_texts, time_list = process_text(result)
                st.session_state.vectorstore = get_vectorstore(grouped_texts, time_list)

            st.write("NOW YOU CAN START CHATTING")


    if user_question:
      chain = get_conversation(st.session_state.vectorstore)

      with st.chat_message('user'):
            st.markdown(user_question)
      st.session_state.message.append({'role': "user", "content": user_question})

      with st.chat_message('assistant'):
                        chain_answer = chain({'question':user_question, "chat_history": []}, return_only_outputs=True)
                        response = chain_answer['answer']
                        st.markdown(response)

      st.session_state.message.append({"role": "assistant", "content": response})



if __name__ == '__main__':
    main()
