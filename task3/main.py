# main.py
from fastapi import FastAPI, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session

from database import Base, engine, get_db
import models
import crud
from schemas import Student, StudentCreate, StudentUpdate

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Student CRUD API (modular)")


@app.get("/")
def read_root():
    return {"message": "Student CRUD API with SQLAlchemy, Pydantic & modular structure is running!"}


# Create
@app.post("/students/", response_model=Student, status_code=201)
def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    return crud.create_student(db, student)


# Read all
@app.get("/students/", response_model=List[Student])
def get_students(db: Session = Depends(get_db)):
    return crud.get_students(db)


# Read one
@app.get("/students/{student_id}", response_model=Student)
def get_student(student_id: int, db: Session = Depends(get_db)):
    student = crud.get_student(db, student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


# Update
@app.put("/students/{student_id}", response_model=Student)
def update_student(student_id: int, updated_data: StudentUpdate, db: Session = Depends(get_db)):
    student = crud.update_student(db, student_id, updated_data)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


# Delete
@app.delete("/students/{student_id}", status_code=204)
def delete_student(student_id: int, db: Session = Depends(get_db)):
    success = crud.delete_student(db, student_id)
    if not success:
        raise HTTPException(status_code=404, detail="Student not found")
    return None
