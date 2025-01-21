# Bruno - Your Personal AI Chatbot

## Project Overview

Bruno Chat-Bot is an interactive chatbot application that enables natural language interaction. It utilizes the Llama model for text generation, Whisper for speech recognition, and Suno's Bark for speech synthesis. The application provides a seamless experience for users to input text or speech, receive text responses, and hear those responses spoken back.

## Installation

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Transformers library by Hugging Face
- Other dependencies as specified in `requirements.txt`

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/personal-chatbot.git
   cd personal-chatbot
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Models**

   - Llama model for text generation
   - Whisper model for speech-to-text
   - Suno's Bark model for text-to-speech

## Running the Application

1. **Launch the App**

   ```bash
   streamlit run app.py
   ```

2. **Access the App**

   - Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Project Structure

- **`app.py`**: The main application file.
- **`llm_chains.py`**: Contains the logic for loading and using the Llama model.
- **`audio_handler.py`**: Handles speech-to-text transcription using Whisper.
- **`requirements.txt`**: Lists all the Python packages required to run the application.

## Code Explanation

- **Audio Input Handling**: Uses `audio_recorder` to capture user speech, which is then transcribed to text using Whisper.
- **Text Generation**: The transcribed text or direct text input is processed by the Llama model to generate a response.
- **Speech Synthesis**: The generated text response is converted to speech using Suno's Bark model and played back to the user.

## Customization

- **Models**: You can replace the models with different versions or alternatives by updating the model loading sections in the code.
- **Parameters**: Adjust model parameters in the respective loading functions for fine-tuning the behavior.

## Limitations and Issues

- **Performance**: The application may require significant computational resources, especially for real-time processing.
- **Dependencies**: Ensure all models and libraries are correctly installed and compatible.

## Contributions

- **Contribute**: Contributions are welcome! Please fork the repository and submit a pull request.




