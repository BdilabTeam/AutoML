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