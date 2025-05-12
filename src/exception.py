import sys
from logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exec_tb=error_detail.exc_info()
    file_name=exec_tb.tb_frame.f_code.co_filename
    error_message=f"Error Occured in python for script {file_name} line number {exec_tb.tb_lineno} error message {error}"
    return error_message

class custom_exception(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)


    def __Str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide By Zero")
        raise custom_exception(e,sys)