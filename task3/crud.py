# crud.py
from sqlalchemy.orm import Session
from models import StudentModel
from schemas import StudentCreate, StudentUpdate


def create_student(db: Session, student: StudentCreate) -> StudentModel:
    db_student = StudentModel(**student.dict())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    return db_student


def get_students(db: Session):
    return db.query(StudentModel).all()


def get_student(db: Session, student_id: int):
    return db.query(StudentModel).filter(StudentModel.id == student_id).first()


def update_student(db: Session, student_id: int, updated_data: StudentUpdate):
    student = get_student(db, student_id)
    if student is None:
        return None

    student.name = updated_data.name
    student.age = updated_data.age
    student.grade = updated_data.grade

    db.commit()
    db.refresh(student)
    return student


def delete_student(db: Session, student_id: int) -> bool:
    student = get_student(db, student_id)
    if student is None:
        return False

    db.delete(student)
    db.commit()
    return True
