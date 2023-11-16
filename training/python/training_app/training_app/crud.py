import models, schemas
from sqlalchemy.orm import Session


def create_training_project(db: Session, training_project: schemas.TrainingProjectCreate) -> models.TrainingProject:
    db_training_project = models.TrainingProject(**training_project.__dict__)
    db.add(db_training_project)
    db.commit()
    db.refresh(db_training_project)
    return db_training_project


def get_training_project(db: Session, training_project_id: int) -> models.TrainingProject:
    return db.query(models.TrainingProject).filter(models.TrainingProject.id == training_project_id).one()

def update_training_project(db: Session, training_project: schemas.TrainingProjectUpdate, training_project_id: int) -> models.TrainingProject:
    db_training_project_update = get_training_project(training_project_id)
    if db_training_project_update:
        db_training_project_update.name = training_project.name
        db_training_project_update.task_type = training_project.task_type
        db_training_project_update.is_automatic = training_project.is_automatic
        db_training_project_update.model_name_or_path = training_project.model_name_or_path
        db_training_project_update.data_name_or_path = training_project.data_name_or_path
    db.commit()
    db.close()
    return db_training_project_update