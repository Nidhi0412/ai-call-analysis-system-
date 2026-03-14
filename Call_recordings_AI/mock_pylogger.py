#!/usr/bin/env python3
"""
Mock Pylogger
=============
Simple mock logger for testing when pylogger is not available
"""

import logging
import os
from datetime import datetime

class MockPylogger:
    """Mock pylogger for testing"""
    
    def __init__(self, log_dir="/tmp", app_name="test"):
        self.log_dir = log_dir
        self.app_name = app_name
        
        # Set up basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(app_name)
    
    def log_it(self, data):
        """Mock log_it method"""
        log_type = data.get("logType", "info")
        prefix = data.get("prefix", "unknown")
        log_data = data.get("logData", {})
        
        message = f"[{prefix}] {log_data.get('message', 'No message')}"
        
        if log_type == "error":
            self.logger.error(message)
        elif log_type == "warning":
            self.logger.warning(message)
        elif log_type == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)
        
        # Print to console for testing
        print(f"[{log_type.upper()}] {message}")

# Create a callable function that returns a MockPylogger instance
def pylogger(log_dir="/tmp", app_name="test"):
    """Mock pylogger function that returns a MockPylogger instance"""
    return MockPylogger(log_dir, app_name) 