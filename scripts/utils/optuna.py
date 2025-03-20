from dataclasses import dataclass
import optuna
import shutil
import os

@dataclass
class ArchiveBestModelCallback:
    out_path: str
    out_path_scratch: str

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        best_scratch = os.path.join(self.out_path_scratch,'model-best')
        best_stable = os.path.join(self.out_path,'model-best')
        if best_stable == best_scratch:
            return # Nothing to do.
        if ((study.direction == optuna.study.StudyDirection.MAXIMIZE
            and study.best_value <= trial.value) or
            (study.direction == optuna.study.StudyDirection.MINIMIZE
             and study.best_value >= trial.value)):
            shutil.copytree(best_scratch, best_stable)