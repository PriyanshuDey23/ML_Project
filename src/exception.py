import sys # Any exception that is getting controled, this library will have that information
import logging

# where ever an exception get raised 
def error_message_detail(error,error_detail:sys):  # error , along with error message stored in sys
    _,_,exc_tb=error_detail.exc_info() # First 2 are not important 3rd -># exc_tb:- On which line or file the exception has occured
    file_name=exc_tb.tb_frame.f_code.co_filename # We will the error file name  
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]" .format(
        file_name,exc_tb.tb_lineno,str(error)       
    )
    return error_message 

class CustomException(Exception):      # Inheriting the parent exception
    def __init__(self,error_message,error_detail:sys):  # 
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):       # We will get the error message(Printing)
        return self.error_message

# When ever we get an exception, log it with logger file use log.info to put it inside the file

