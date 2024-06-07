from abc import ABC, abstractmethod
from typing import Dict

class TaskManager(ABC):
    def __init__(self, name: str, task_config: Dict, model, logger):
        self.name = name
        self.task_config = task_config
        self.model = model
        self.logger = logger

    @abstractmethod
    def run(self, batch_size: int, output_folder: str):
        pass