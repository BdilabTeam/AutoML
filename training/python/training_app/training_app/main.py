from database import (
    Base,
    engine
)


# 生成schema
# models.TrainingProject.metadata.create_all(engine)
Base.metadata.create_all(engine)
