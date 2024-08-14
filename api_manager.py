# api_manager.py
import os

class APIManager:
    def __init__(self, api_key=None):
        if api_key:
            self.set_api_key(api_key)

    def set_api_key(self, key):
        os.environ["UPSTAGE_API_KEY"] = key
        self.upstage_api_key = key