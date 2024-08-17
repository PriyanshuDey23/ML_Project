import logging # For tracking everything
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # Data along with time
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE) # Curent work. Direct., folder will get created in CWD , every file will start will logs along with LOG FILE 
os.makedirs(log_path,exist_ok=True) # Keep on appending the files in the folder 

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE) 

logging.basicConfig(
    filename=LOG_FILE_PATH,           # Where we want to save it
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s %(message)s",  # How my message will get printed
    level=logging.INFO,       # with the help of info I am going to rint the specific message
)
