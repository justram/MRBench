import os
from typing import List
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from src.tasks.task_manager import TaskManager
from src.tasks.retrieval import RetrievalTaskManager
from src.tasks.classification import ClassificationTaskManager
from src.utils.logger import Logger

class EVAL:
    def __init__(self, config: DictConfig):
        self.config = config
        self.output_folder = config.output_folder
        self.cache_folder = config.cache_folder
        self.batch_size = config.batch_size
        self.model = self.load_model(config.model_args)
        self.logger = Logger(log_dir=config.log_dir)
        self.ensure_output_folder()
        self.ensure_cache_folder()
        self.tasks = self.load_tasks()

    def ensure_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.logger.log(f"Output folder created at {self.output_folder}")

    def ensure_cache_folder(self):
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.logger.log(f"Cache folder created at {self.cache_folder}")

    def load_model(self, model_args):
        # Load and return the model based on model_args
        if model_args.model_type == 'clip':
            from src.models.hgf import CLIP
            return CLIP(model_args)
        raise ValueError(f"Unknown model: {model_args.model_name_or_path}")

    def load_tasks(self) -> List[TaskManager]:
        task_objects = []
        for task in self.config.tasks:
            task_type = task['type']
            if not task_type:
                raise ValueError(f"Unknown task type: {task_type}")
            
            config_path = to_absolute_path(f"config/task/{task_type}/{task['dataset_name']}.yaml")
            self.logger.log(f"Loading task config from {config_path}")
            if not os.path.exists(config_path):
                self.logger.log(f"Config file {config_path} does not exist")
                raise FileNotFoundError(f"Config file {config_path} does not exist")
            
            task_config = OmegaConf.load(config_path)
            callbacks = task_config.get('callbacks', {})
            if task_type == "retrieval":
                task_objects.append(
                    RetrievalTaskManager(
                        name=task_config['name'],
                        task_config=task_config,
                        model=self.model,
                        logger=self.logger,
                        cache_folder=self.cache_folder,
                        callbacks=callbacks
                    )
                )
            elif task_type == "classification":
                task_objects.append(
                    ClassificationTaskManager(
                        name=dataset_name,
                        task_config=task_config,
                        model=self.model,
                        logger=self.logger,
                        cache_folder=self.cache_folder,
                        callbacks=callbacks
                    )
                )
        return task_objects

    def run(self):
        for task in self.tasks:
            task.run(self.batch_size, self.output_folder)

@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    evaluator = EVAL(config)
    evaluator.run()

if __name__ == "__main__":
    main()
