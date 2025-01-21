from prompt import prompt_template
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import yaml
import os
import logging
from dotenv import load_dotenv
HUGGINGFACEHUB_API_TOKEN="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"
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

os.environ['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'] = os.getenv("HUGGINGFACEHUB_API", "")
if not os.environ['HUGGINGFACEHUB_API_TOKEN']:
    logger.warning("HUGGINGFACEHUB_API_TOKEN is not set. Ensure it's available in the .env file.")

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("Config file 'config.yaml' not found.")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing 'config.yaml': {e}")
    raise

def create_llm(model_path=config["model_path"]["chat_model_llama"], model_type=config["model_type"], model_config=config["model_config"]):
    try:
        llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
        logger.info("LLM created successfully.")
        return llm
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        raise

def create_prompt_from_template(template):
    return PromptTemplate(input_variables=["prompt"], template=template)

def load_normal_chain():
    return chatChain()

class chatChain:
    def __init__(self):
        try:
            self.llm = create_llm()
            self.chat_prompt = create_prompt_from_template(prompt_template)
            # Uncomment the following line if memory is needed
            # self.llm_chain = LLMChain(llm=self.llm, prompt=self.chat_prompt, memory=ConversationBufferWindowMemory(k=1))
            self.llm_chain = self.chat_prompt | self.llm
            logger.info("Chat chain initialized.")
        except Exception as e:
            logger.error(f"Error initializing chatChain: {e}")
            raise

    def run(self, user_input):
        try:
            logger.info(f"Processing input: {user_input}")
            return self.llm_chain.invoke(user_input)
        except Exception as e:
            logger.error(f"Error during chain execution: {e}")
            raise
