import speech_recognition as sr
from gtts import gTTS  # Google text to speech
import os
import transformers  # Courtesy of Hugging Face for text generation
import datetime
import numpy as np


class ChatBot():
    def __init__(self, name):
        print("---", name, "is running" " ---")
        self.name = name

    def speech_to_text(self):
        r = sr.Recognizer()  # Create a new speech recognition instance
        with sr.Microphone() as mic:  # Create a new microphone instance utilizing the device mic
            print("Say something...")
            # Dynamic energy threshold adjustment
            r.adjust_for_ambient_noise(mic, duration=1)
            # Record from audio instance into audioData instance
            audio = r.listen(mic)
        try:
            # Leverages the Google Speech Recognition API on audioData
            self.text = r.recognize_google(audio)
            print(" You said: " + self.text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(
                "Could not request results from Google Speech Recognition service; {0}".format(e))

    @staticmethod
    def text_to_speech(text):  # Converting text response to .mp3 file for audio playback
        print("Robot: ", text)
        # Basic Google Text to Speech without preprocessor funcs
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("robot.mp3")  # Save .mp3 audio response
        # Playback audio file (afplay is Mac specific)
        os.system("afplay robot.mp3")
        os.remove("robot.mp3")  # Remove audio after playback

    def wake_up(self, text):
        # Identify robot name in audioData to begin conversation "Hey Robot"
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        # Static datetime response to test functionality
        return datetime.datetime.now().time().strftime('It is currently %H:%M')


if __name__ == "__main__":
    ai = ChatBot(name="robot")  # Determine name for wake + text output
    nlp = transformers.pipeline(
        "conversational", model="microsoft/DialoGPT-medium")  # Using Microsoft pretrained response generation model
    # Enable data parallelism to split data into n partitions
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    while True:
        ai.speech_to_text()
        if ai.wake_up(ai.text) is True:  # Set detection of name for conversation start
            res = "Hello I am Bot, what can I do for you?"
        elif "time" in ai.text:  # Utilize static datetime for testing
            res = ai.action_time()
        # Determine static gratification responses for test
        elif any(i in ai.text for i in ["thank", "thanks", "thank you", "I appreciate it"]):
            res = np.random.choice(
                ["you're welcome!", "anytime!",
                 "no problem!", "absolutely!",
                 "no worries!", "If you need anything else, just ask!"])
        else:  # If no static response, utilize Microsoft DialoGPT model
            chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
            res = str(chat)
            res = res[res.find("bot >> ")+6:].strip()
        ai.text_to_speech(res)
