from ..models import taining_project_model
from sqlalchemy.orm import Session


def create_training_project(session: Session, training_project: taining_project_model.TrainingProject) -> taining_project_model.TrainingProject:
    session.add(training_project)
    session.flush()
    return training_project

def get_training_project(session: Session, training_project_id: int) -> taining_project_model.TrainingProject:
    return session.query(taining_project_model.TrainingProject).filter(taining_project_model.TrainingProject.id == training_project_id).one()

def update_training_project(session: Session, training_project: taining_project_model.TrainingProject) -> taining_project_model.TrainingProject:
    session.merge(training_project)
    return training_project

def delete_training_project(session: Session, training_project_id: int):
    training_project = session.query(taining_project_model.TrainingProject).filter(taining_project_model.TrainingProject.id == training_project_id).first()
    session.delete(training_project)