#!/usr/bin/env python3
"""
Mock Pylogger
=============

Simple mock implementation of pylogger for local model services
"""

import logging
import os
from datetime import datetime

class MockPylogger:
    """Mock implementation of pylogger"""
    
    def __init__(self, log_dir="/tmp", service_name="local_models"):
        self.log_dir = log_dir
        self.service_name = service_name
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(log_dir, f"{service_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def log_it(self, data):
        """Log data in the expected format"""
        try:
            log_type = data.get('logType', 'info')
            prefix = data.get('prefix', 'general')
            log_data = data.get('logData', {})
            
            message = f"[{prefix}] {log_data}"
            
            if log_type == 'error':
                self.logger.error(message)
            elif log_type == 'warning':
                self.logger.warning(message)
            elif log_type == 'debug':
                self.logger.debug(message)
            else:
                self.logger.info(message)
                
        except Exception as e:
            self.logger.error(f"Error in log_it: {e}")

# Create a global instance
pylogger = MockPylogger()

def get_pylogger(log_dir="/tmp", service_name="local_models"):
    """Get a pylogger instance"""
    return MockPylogger(log_dir, service_name) 