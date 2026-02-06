from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from src.database import get_db
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/media", StaticFiles(directory="media"), name="media")
# import os
# print(os.path.exists("media/user_profile"))

from src.endpoints.auth.self_attendance import router as self_attendance_router
app.include_router(self_attendance_router)

from src.endpoints.api.v1.univercity_management import router as univercity_management_router
app.include_router(univercity_management_router)

from src.endpoints.api.v1.hostel_management import router as hostel_management_router
app.include_router(hostel_management_router)

from src.endpoints.api.v1.student_management import router as register_student_router
app.include_router(register_student_router)

from src.endpoints.auth.login import router as login_router
app.include_router(login_router)

from src.endpoints.api.v1.user_management import router as user_management_router
app.include_router(user_management_router)

from src.endpoints.api.v1.class_management import router as class_management_router
app.include_router(class_management_router)

from src.endpoints.api.v1.student_attendance_session import router as student_attendance_session
app.include_router(student_attendance_session)

# from src.endpoints.api.subject_management import router as subject_management_router
# app.include_router(subject_management_router)

# from src.endpoints.api.allocation import router as allocation_router
# app.include_router(allocation_router)

# from src.endpoints.api.period_management import router as period_management_router
# app.include_router(period_management_router)


