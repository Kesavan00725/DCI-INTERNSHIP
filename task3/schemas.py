# schemas.py
from pydantic import BaseModel


class StudentBase(BaseModel):
    name: str
    age: int
    grade: str


class StudentCreate(StudentBase):
    """Schema for creating a student"""
    pass


class StudentUpdate(StudentBase):
    """Schema for updating a student"""
    pass


class Student(StudentBase):
    """Schema for reading a student (includes id)"""
    id: int

    class Config:
        orm_mode = True   # Important to work with SQLAlchemy models
