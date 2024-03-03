from typing import List
from sqlalchemy import desc
from sqlalchemy.orm import Session
from ..models.experiment import Experiment


def create_experiment(session: Session, experiment: Experiment) -> Experiment:
    session.add(experiment)
    session.flush()
    return experiment

def get_experiment(session: Session, experiment_name: str) -> Experiment:
    return session.query(Experiment).filter(Experiment.experiment_name == experiment_name).first()

def get_all_experiments(session: Session) -> List[Experiment]:
    return session.query(Experiment).order_by(desc(Experiment.id)).all()

def update_experiment(session: Session, experiment:Experiment) -> Experiment:
    session.merge(experiment)
    return experiment

def delete_experiment(session: Session, experiment_name: str):
    experiment = session.query(Experiment).filter(Experiment.experiment_name == experiment_name).first()
    session.delete(experiment)