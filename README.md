This simple text generation chatbot is designed to accept user voice input, and provide a generated audio response.


The backbone of this code is running on the [Microsoft DialoGPT-Medium Pretrained Response Generation Model.](https://github.com/microsoft/DialoGPT)

In addition, we implement the [Google Speech Recognition API](https://cloud.google.com/speech-to-text) along with [Google Text to Speech](https://gtts.readthedocs.io/en/latest/), and [Hugging Face Transformers for Natural Language Processing](https://huggingface.co/docs/transformers/index).

Firstly, inside a speech_to_text method, the application accepts user audio input through the specified device microphone:
```python
r = sr.Recognizer() # Create a new speech recognition instance
with sr.Microphone() as mic: # Create a new microphone instance utilizing the device mic
r.adjust_for_ambient_noise(mic, duration=1) # Dynamic energy threshold adjustment
audio = r.listen(mic) # Record from audio instance into audioData instance
```

Next, we have to leverage the Google Speech Recognition API on the audioData to ensure the data can be parsed:
```python
self.text = r.recognize_google(audio)
```

In a text_to_speech method, we convert the generated text response into a .mp3 file for audio playback utilizing Google Text to Speech. We are implementing the basic capabilities of gTTS, without preprocessor functions:
```python
speaker = gTTS(text=text, lang="en", slow=False)
speaker.save("robot.mp3")
os.system("afplay robot.mp3") # Code here is OS specific, 'afplay' for MacOS and 'start' for Windows
os.remove("robot.mp3") # Remove the audio file after output to regain storage
```

Beginning the conversation is dependant on a matching to the robot name variable in the initial statement, similar to "Hey Alexa":
```python
return True if self.name in text.lower() else False
```
Without waking the bot, Google Speech Recognition will not proceed:
```python
except sr.UnknownValueError:
  print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
  print("Could not request results from Google Speech Recognition service; {0}".format(e))
```
Inside of __main__ we instantiate the Microsoft DialoGPT-Medium Pretrained Response Generation Model and enable data parallelism to split data in n partitions:
```python
ai = ChatBot(name="robot")
nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
```
In the full code you can see static responses being utilized for development and testing.
The Microsoft Model will be utilized, assuming those specific parameters (which can be removed) are not filled, calling our conversational pipeline with specific end of string token ID:
```python
chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
```
NOTE: The token ID can be abstracted by adding the AutoTokenizer class, encoding the user input and eos token, and returning PyTorch tensor objects instead of Python integers:
```python
tokenizer = AutoTokenizer.from_pretrained(<model>)
...
foo = tokenizer.encode(input("bar") + tokenizer.eos_token, return_tensors='pt')
```
