import uuid
from datetime import datetime, time
from enum import Enum
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text, ForeignKey, 
    Time, Date, Enum as SQLEnum, Float, JSON as SQLJSON, 
    UniqueConstraint, CheckConstraint, Index
)
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class UserRole(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"
    STAFF = "staff"
    HOD = "hod"

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class AttendanceStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"
    EXCUSED = "excused"
    HALF_DAY = "half_day"

class ScheduleType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALTERNATE = "alternate"
    CUSTOM = "custom"

class DayOfWeek(str, Enum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"

class LoginType(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"
    FACE_RECOGNITION = "face_recognition"


"""
manage university/school/hostel   
"""
class University(Base):
    __tablename__ = "universities"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    logo_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    schools = relationship("School", back_populates="university", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<University {self.name}>"

class School(Base):
    __tablename__ = "schools"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    university_id = Column(String(36), ForeignKey("universities.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String(50), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    address = Column(Text, nullable=True)
    lat = Column(String(25), nullable=True)
    long = Column(String(25), nullable=True)
    is_active = Column(Boolean, default=False)
    
    # Relationships
    university = relationship("University", back_populates="schools")
    config = relationship("SchoolConfig", back_populates="school", uselist=False, cascade="all, delete-orphan")
    hostels = relationship("Hostel", back_populates="school", cascade="all, delete-orphan")
    users = relationship("User", back_populates="school", cascade="all, delete-orphan")
    classes = relationship("Class", back_populates="school", cascade="all, delete-orphan")
    subjects = relationship("Subject", back_populates="school", cascade="all, delete-orphan")
    holidays = relationship("Holiday", back_populates="school", cascade="all, delete-orphan")
    special_events = relationship("SpecialEvent", back_populates="school", cascade="all, delete-orphan")
    enrollments = relationship("Enrollment", back_populates="school", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<School {self.name}>"

class SchoolConfig(Base):
    __tablename__ = "school_configs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False, unique=True)
    academic_year = Column(String(20), nullable=False)
    schedule_type = Column(SQLEnum(ScheduleType), default=ScheduleType.WEEKLY)
    default_periods_per_day = Column(Integer, default=8)
    period_duration = Column(Integer, default=240)
    break_duration = Column(Integer, default=60)
    school_start_time = Column(Time, default=time(10, 0))
    school_end_time = Column(Time, default=time(15, 0))
    is_half_day_allowed = Column(Boolean, default=True)
    half_day_periods = Column(Integer, default=4)
    enable_automatic_schedule = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    lat = Column(String(20), nullable=True)
    long = Column(String(20), nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    school = relationship("School", back_populates="config")

    def __repr__(self):
        return f"<SchoolConfig {self.academic_year}>"

class Hostel(Base):
    __tablename__ = "hostels"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    room = Column(JSON, default={})
    address = Column(Text)
    capacity = Column(Integer)
    warden_id = Column(String(36), ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    school = relationship("School", back_populates="hostels")
    warden = relationship("User", foreign_keys=[warden_id], backref="warden_of_hostels")
    enrollments = relationship("Enrollment", back_populates="hostel", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_hostel_school', 'school_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Hostel {self.name}>"


##### user management
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    fullname = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(20))
    password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    department = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    school = relationship("School", back_populates="users")
    user_details = relationship("UserDetails", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    # Teacher relationships
    class_teacher_allocations = relationship("ClassTeacherAllocation", back_populates="teacher", cascade="all, delete-orphan")
    period_timetables = relationship("PeriodTimetable", back_populates="teacher", cascade="all, delete-orphan")
    attendance_sessions = relationship("AttendanceSession", back_populates="teacher", cascade="all, delete-orphan")
    
    # Attendance and leave
    attendance_records = relationship("UserAttendance", back_populates="user", cascade="all, delete-orphan")
    leave_requests = relationship("LeaveRequest", back_populates="user", foreign_keys="[LeaveRequest.user_id]", cascade="all, delete-orphan")
    approved_leave_requests = relationship("LeaveRequest", back_populates="approver", foreign_keys="[LeaveRequest.approved_by]", cascade="all, delete-orphan")
    
    # Other relationships
    created_students = relationship("Student", back_populates="creator", foreign_keys="[Student.created_by]", cascade="all, delete-orphan")
    created_classes = relationship("Class", back_populates="creator", foreign_keys="[Class.created_by]", cascade="all, delete-orphan")
    recorded_attendance = relationship("StudentAttendance", back_populates="recorder", foreign_keys="[StudentAttendance.recorded_by]", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.email} ({self.role})>"

class UserDetails(Base):
    __tablename__ = "user_details"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )

    dob = Column(Date, nullable=True)
    gender = Column(SQLEnum(Gender))
    address = Column(Text)
    photo_url = Column(String(500))
    face_encoding_data = Column(JSON)
    max_periods_per_day = Column(Integer, default=6)
    max_periods_per_week = Column(Integer, default=30)

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    user = relationship("User", back_populates="user_details")

    def __repr__(self):
        return f"<UserDetails {self.user_id}>"

####### student with enrollment
class Student(Base):
    __tablename__ = "students"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    dob = Column(Date, nullable=False)
    gender = Column(SQLEnum(Gender), nullable=False)
    email = Column(String(255), unique=True)
    phone = Column(String(20), nullable=True)
    address = Column(Text, nullable=True)
    roll_number = Column(String(20), nullable=True)
    academic_year = Column(String(9), nullable=False)
    # face_image_url = Column(String(255), nullable=True)
    face_encoding_data = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True)
    created_by = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    school = relationship("School")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_students")
    enrollments = relationship("Enrollment", back_populates="student", cascade="all, delete-orphan")
    attendance_records = relationship("StudentAttendance", back_populates="student", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint("school_id", "roll_number", "academic_year", name="uq_student_roll"),
        Index("idx_student_school", "school_id", "is_active"),
    )

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def __repr__(self):
        return f"<Student {self.roll_number}: {self.full_name}>"

class Enrollment(Base):
    __tablename__ = "enrollments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(String(36), ForeignKey("classes.id", ondelete="CASCADE"), nullable=False)
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    hostel_id = Column(String(36), ForeignKey("hostels.id", ondelete="CASCADE"), nullable=True)
    academic_year = Column(String(9), nullable=False)
    enrollment_date = Column(Date, nullable=False, default=datetime.now().date())
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    student = relationship("Student", back_populates="enrollments")
    class_ = relationship("Class", back_populates="enrollments")
    school = relationship("School", back_populates="enrollments")
    hostel = relationship("Hostel", back_populates="enrollments")
    
    __table_args__ = (
        UniqueConstraint('student_id', 'class_id', 'academic_year', name='unique_student_class_year'),
        Index('idx_enrollment_class', 'class_id', 'academic_year'),
        Index('idx_enrollment_student', 'student_id', 'status'),
    )
    
    def __repr__(self):
        return f"<Enrollment: Student {self.student_id} in Class {self.class_id}>"


###### period Management #####
class Class(Base):
    __tablename__ = "classes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(50), nullable=False)
    section = Column(String(10), nullable=False)
    room_number = Column(String(20), nullable=True)
    current_strength = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    academic_year = Column(String(9), nullable=False)
    created_by = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    school = relationship("School", back_populates="classes")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_classes")
    enrollments = relationship("Enrollment", back_populates="class_", cascade="all, delete-orphan")
    class_teacher_allocations = relationship("ClassTeacherAllocation", back_populates="class_", cascade="all, delete-orphan")
    class_schedules = relationship("ClassSchedule", back_populates="class_", cascade="all, delete-orphan")
    attendance_sessions = relationship("AttendanceSession", back_populates="class_", cascade="all, delete-orphan")
    student_attendance = relationship("StudentAttendance", back_populates="class_", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('school_id', 'name', 'section', 'academic_year', name='unique_class_school_year'),
        Index('idx_class_school', 'school_id', 'is_active'),
    )
    
    @property
    def full_name(self):
        return f"{self.name} - {self.section}"
    
    def __repr__(self):
        return f"<Class {self.full_name}>"

####### holiday and Special Event ##########
class Holiday(Base):
    __tablename__ = "holidays"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    name = Column(String(200), nullable=False)
    holiday_type = Column(String(50), default="general")
    description = Column(Text)
    affects_classes = Column(Boolean, default=True)
    affects_staff = Column(Boolean, default=True)
    is_makeup_day = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    school = relationship("School", back_populates="holidays")
    
    __table_args__ = (
        UniqueConstraint('school_id', 'date', name='unique_school_holiday_date'),
        Index('idx_holiday_date', 'date'),
        Index('idx_holiday_school', 'school_id', 'date'),
    )
    
    def __repr__(self):
        return f"<Holiday {self.date}: {self.name}>"

class SpecialEvent(Base):
    __tablename__ = "special_events"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(200), nullable=False)
    event_type = Column(String(50), nullable=False)
    description = Column(Text)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    start_time = Column(Time)
    end_time = Column(Time)
    affected_classes = Column(JSON)
    routine_change_type = Column(String(50), default="no_classes")
    modified_schedule = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    school = relationship("School", back_populates="special_events")
    
    def __repr__(self):
        return f"<SpecialEvent {self.title}>"

##### student attendance ##########
class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    period_timetable_id = Column(String(36), ForeignKey("period_timetable.id", ondelete="CASCADE"), nullable=True)
    session_date = Column(Date, nullable=False)
    scheduled_start = Column(Time)
    scheduled_end = Column(Time)
    teacher_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(String(36), ForeignKey("classes.id", ondelete="CASCADE"), nullable=False)
    subject_id = Column(String(36), ForeignKey("subjects.id", ondelete="CASCADE"), nullable=True)
    
    # Status
    status = Column(String(20), default="scheduled")
    attendance_method = Column(String(20), default="manual")
    
    # Statistics
    total_students = Column(Integer, default=0)
    present_count = Column(Integer, default=0)
    absent_count = Column(Integer, default=0)
    late_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    period_timetable = relationship("PeriodTimetable", back_populates="attendance_sessions")
    teacher = relationship("User", back_populates="attendance_sessions")
    class_ = relationship("Class", back_populates="attendance_sessions")
    subject = relationship("Subject", back_populates="attendance_sessions")
    student_attendance = relationship("StudentAttendance", back_populates="attendance_session", cascade="all, delete-orphan")
    capture_logs = relationship("AttendanceCaptureLog", back_populates="attendance_session", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('period_timetable_id', 'session_date', name='unique_session_period_date'),
        Index('idx_session_date', 'session_date'),
        Index('idx_session_teacher', 'teacher_id', 'session_date'),
    )
    
    def __repr__(self):
        return f"<AttendanceSession {self.session_date}>"

class AttendanceCaptureLog(Base):
    __tablename__ = "attendance_capture_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    attendance_session_id = Column(String(36), ForeignKey("attendance_sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    image_url = Column(String(500))
    recognition_confidence = Column(Float)
    detected_faces = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    attendance_session = relationship("AttendanceSession", back_populates="capture_logs")
    
    def __repr__(self):
        return f"<AttendanceCaptureLog for Session {self.attendance_session_id}>"

class StudentAttendance(Base):
    __tablename__ = "student_attendance"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    class_id = Column(String(36), ForeignKey("classes.id", ondelete="CASCADE"), nullable=False)
    attendance_session_id = Column(String(36), ForeignKey("attendance_sessions.id", ondelete="CASCADE"))
    attendance_date = Column(Date, nullable=False)
    status = Column(SQLEnum(AttendanceStatus), nullable=False)
    is_face_detected = Column(Boolean, default=False)
    is_late = Column(Boolean, default=False)
    late_minutes = Column(Integer, default=0)
    remarks = Column(Text)
    recorded_by = Column(String(36), ForeignKey("users.id"))
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    student = relationship("Student", back_populates="attendance_records")
    class_ = relationship("Class", back_populates="student_attendance")
    attendance_session = relationship("AttendanceSession", back_populates="student_attendance")
    recorder = relationship("User", foreign_keys=[recorded_by], back_populates="recorded_attendance")
    
    __table_args__ = (
        UniqueConstraint('student_id', 'attendance_date', 'attendance_session_id', name='unique_student_date_session'),
        Index('idx_attendance_date', 'attendance_date'),
        Index('idx_student_attendance', 'student_id', 'attendance_date'),
        Index('idx_class_attendance', 'class_id', 'attendance_date'),
    )
    
    def __repr__(self):
        return f"<StudentAttendance {self.student_id} - {self.attendance_date}>"


############# user Attendance #######
class UserAttendance(Base):
    __tablename__ = "user_attendance"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    attendance_date = Column(Date, nullable=False)
    login_time = Column(Time, nullable=False)
    logout_time = Column(Time)
    status = Column(String(20), default="present")
    login_type = Column(SQLEnum(LoginType), default=LoginType.FACE_RECOGNITION)
    is_face_detected = Column(Boolean, default=False)
    late_minutes = Column(Integer, default=0)
    remarks = Column(Text)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="attendance_records")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'attendance_date', name='unique_user_date'),
        Index('idx_user_attendance_date', 'user_id', 'attendance_date'),
    )
    
    def __repr__(self):
        return f"<UserAttendance {self.user_id} - {self.attendance_date}>"


########################### End Here ##############################

#Extra model
class Subject(Base):
    __tablename__ = "subjects"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    school_id = Column(String(36), ForeignKey("schools.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    color_code = Column(String(7), default='#3498db')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    school = relationship("School", back_populates="subjects")
    class_teacher_allocations = relationship("ClassTeacherAllocation", back_populates="subject", cascade="all, delete-orphan")
    period_timetables = relationship("PeriodTimetable", back_populates="subject", cascade="all, delete-orphan")
    attendance_sessions = relationship("AttendanceSession", back_populates="subject", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_subject_school', 'school_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Subject {self.name}>"

class ClassTeacherAllocation(Base):
    __tablename__ = "class_teacher_allocations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    class_id = Column(String(36), ForeignKey("classes.id", ondelete="CASCADE"), nullable=True)
    teacher_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    subject_id = Column(String(36), ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False)
    academic_year = Column(String(9), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    class_ = relationship("Class", back_populates="class_teacher_allocations")
    teacher = relationship("User", back_populates="class_teacher_allocations")
    subject = relationship("Subject", back_populates="class_teacher_allocations")
    
    __table_args__ = (
        UniqueConstraint('class_id', 'subject_id', 'academic_year', name='unique_class_subject_year'),
        UniqueConstraint('class_id', 'teacher_id', 'academic_year', name='unique_class_teacher_year'),
        Index('idx_allocation_teacher', 'teacher_id', 'academic_year'),
    )
    
    def __repr__(self):
        return f"<ClassTeacherAllocation: Teacher {self.teacher_id} for Class {self.class_id}>"

class ClassSchedule(Base):
    __tablename__ = "class_schedules"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    class_id = Column(String(36), ForeignKey("classes.id", ondelete="CASCADE"), nullable=False)
    schedule_type = Column(SQLEnum(ScheduleType), nullable=False)
    periods_per_day = Column(Integer, nullable=False)
    effective_from = Column(Date, nullable=True)
    effective_to = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Day-specific periods
    monday_periods = Column(Integer, default=0)
    tuesday_periods = Column(Integer, default=0)
    wednesday_periods = Column(Integer, default=0)
    thursday_periods = Column(Integer, default=0)
    friday_periods = Column(Integer, default=0)
    saturday_periods = Column(Integer, default=0)
    sunday_periods = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    class_ = relationship("Class", back_populates="class_schedules")
    period_timetable = relationship("PeriodTimetable", back_populates="class_schedule", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_schedule_class', 'class_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<ClassSchedule for Class {self.class_id}>"

class PeriodTimetable(Base):
    __tablename__ = "period_timetable"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    class_schedule_id = Column(String(36), ForeignKey("class_schedules.id", ondelete="CASCADE"), nullable=False)
    
    # Period details
    period_number = Column(Integer, nullable=False)
    day_of_week = Column(SQLEnum(DayOfWeek))
    week_type = Column(String(20))
    month_week = Column(Integer)
    
    # Timing
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    
    # Subject and teacher
    subject_id = Column(String(36), ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False)
    teacher_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    room = Column(String(50))
    
    # Status
    is_break = Column(Boolean, default=False)
    break_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    class_schedule = relationship("ClassSchedule", back_populates="period_timetable")
    subject = relationship("Subject", back_populates="period_timetables")
    teacher = relationship("User", back_populates="period_timetables")
    attendance_sessions = relationship("AttendanceSession", back_populates="period_timetable", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('class_schedule_id', 'period_number', 'day_of_week', 'week_type', 'month_week', 
                        name='unique_period_schedule'),
        CheckConstraint('period_number > 0', name='check_period_positive'),
        Index('idx_timetable_teacher', 'teacher_id', 'day_of_week'),
    )
    
    def __repr__(self):
        return f"<PeriodTimetable Period {self.period_number}>"

class LeaveRequest(Base):
    __tablename__ = "leave_requests"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    leave_type = Column(String(50), nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String(20), default="pending")
    approved_by = Column(String(36), ForeignKey("users.id"))
    approved_at = Column(DateTime)
    rejection_reason = Column(Text)
    need_substitute = Column(Boolean, default=True)
    substitute_assigned = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="leave_requests", foreign_keys=[user_id])
    approver = relationship("User", back_populates="approved_leave_requests", foreign_keys=[approved_by])
    
    __table_args__ = (
        Index('idx_leave_dates', 'start_date', 'end_date'),
        Index('idx_leave_user', 'user_id', 'status'),
    )
    
    def __repr__(self):
        return f"<LeaveRequest {self.user_id} - {self.start_date} to {self.end_date}>"