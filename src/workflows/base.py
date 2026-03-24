from quant_derivatives.utils.logging_config import setup_logging

class BaseWorkflow:
    def __init__(self, command_name: str, args):
        self.logger = setup_logging()
        self.logger.info(f"Initialized {command_name} workflow")
