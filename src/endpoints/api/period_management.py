import uuid
import random
from datetime import time, datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, status, Form, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, not_
from pydantic import BaseModel

from src.database import get_db
from src.models import (
    PeriodTimetable, ClassSchedule, ClassTeacherAllocation,
    Class, User, Subject, School, SchoolConfig,
    DayOfWeek, ScheduleType, UserRole
)
from src.core.utils.customize_response import success_response, error_response

router = APIRouter(
    prefix="/api/period",
    tags=["Period Generation"]
)


# =========================================================
# 🔹 HELPER FUNCTIONS FOR PERIOD GENERATION
# =========================================================

def get_school_config(db: Session, school_id: str) -> SchoolConfig:
    """Get school configuration for a given school ID"""
    school_config = db.query(SchoolConfig).filter(
        SchoolConfig.school_id == school_id
    ).first()
    
    if not school_config:
        raise ValueError(f"School configuration not found for school ID: {school_id}")
    
    return school_config


def calculate_period_timings(
    school_start_time: time,
    school_end_time: time,
    period_duration: int,
    break_duration: int,
    total_periods: int
) -> List[Tuple[time, time]]:
    """
    Calculate start and end times for each period based on school timings.
    
    Logic:
    1. Calculate total available minutes
    2. Subtract break times (breaks after each period except last)
    3. Divide remaining time equally among periods
    4. Adjust for actual period duration
    """
    # Convert times to minutes since midnight
    start_minutes = school_start_time.hour * 60 + school_start_time.minute
    end_minutes = school_end_time.hour * 60 + school_end_time.minute
    
    total_available_minutes = end_minutes - start_minutes
    total_break_minutes = break_duration * (total_periods - 1)  # Breaks between periods
    
    # Time available for teaching
    teaching_minutes = total_available_minutes - total_break_minutes
    
    # Calculate actual period duration (might be less than configured if not enough time)
    actual_period_duration = min(period_duration, teaching_minutes // total_periods)
    
    period_timings = []
    current_time = start_minutes
    
    for period_num in range(1, total_periods + 1):
        period_start = current_time
        period_end = period_start + actual_period_duration
        
        # Convert back to time objects
        start_hour = period_start // 60
        start_minute = period_start % 60
        
        end_hour = period_end // 60
        end_minute = period_end % 60
        
        period_timings.append((
            time(hour=start_hour, minute=start_minute),
            time(hour=end_hour, minute=end_minute)
        ))
        
        # Add break after period (except after last period)
        if period_num < total_periods:
            current_time = period_end + break_duration
        else:
            current_time = period_end
    
    return period_timings


def get_teacher_availability_matrix(
    db: Session,
    school_id: str,
    days_of_week: List[str],
    period_timings: List[Tuple[time, time]]
) -> Dict[str, Dict[str, List[int]]]:
    """
    Create availability matrix for teachers.
    Returns: {teacher_id: {day: [available_period_indices]}}
    """
    teacher_availability = defaultdict(lambda: defaultdict(list))
    
    # Get all teachers in the school
    teachers = db.query(User).filter(
        User.school_id == school_id,
        User.is_active == True,
        User.role.in_([UserRole.TEACHER, UserRole.HOD])  # Only teachers and HODs
    ).all()
    
    # Initialize all teachers as available for all periods on all days
    for teacher in teachers:
        for day in days_of_week:
            teacher_availability[teacher.id][day] = list(range(len(period_timings)))
    
    # Remove periods where teacher already has classes
    existing_periods = db.query(PeriodTimetable).join(
        ClassSchedule, PeriodTimetable.class_schedule_id == ClassSchedule.id
    ).join(
        Class, ClassSchedule.class_id == Class.id
    ).filter(
        Class.school_id == school_id,
        PeriodTimetable.is_active == True,
        PeriodTimetable.day_of_week.in_(days_of_week)
    ).all()
    
    for period in existing_periods:
        teacher_id = period.teacher_id
        day = period.day_of_week.value
        
        # Find which period index this conflicts with based on timing
        for idx, (start_time, end_time) in enumerate(period_timings):
            if (period.start_time == start_time and period.end_time == end_time):
                if idx in teacher_availability[teacher_id][day]:
                    teacher_availability[teacher_id][day].remove(idx)
                break
    
    return teacher_availability


def get_all_subjects_for_school(db: Session, school_id: str) -> List[Subject]:
    """Get all active subjects in the school"""
    return db.query(Subject).filter(
        Subject.school_id == school_id,
        Subject.is_active == True
    ).all()


def get_teacher_subject_assignments(
    db: Session,
    school_id: str,
    classes: List[Class]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get teacher-subject assignments for the school.
    
    Returns two dictionaries:
    1. teacher_subjects: {teacher_id: [subject_id1, subject_id2, ...]}
    2. subject_teachers: {subject_id: [teacher_id1, teacher_id2, ...]}
    
    Logic:
    - First, use explicit ClassTeacherAllocation if available
    - If no explicit allocations, all teachers can teach all subjects
    """
    # Initialize dictionaries
    teacher_subjects = defaultdict(list)
    subject_teachers = defaultdict(list)
    
    # Get all teachers in the school
    all_teachers = db.query(User).filter(
        User.school_id == school_id,
        User.is_active == True,
        User.role.in_([UserRole.TEACHER, UserRole.HOD])
    ).all()
    
    # Get all subjects in the school
    all_subjects = get_all_subjects_for_school(db, school_id)
    all_subject_ids = [subject.id for subject in all_subjects]
    
    # Check if there are any explicit teacher allocations in the school
    has_explicit_allocations = False
    for class_obj in classes:
        allocations_count = db.query(ClassTeacherAllocation).filter(
            ClassTeacherAllocation.class_id == class_obj.id
        ).count()
        
        if allocations_count > 0:
            has_explicit_allocations = True
            break
    
    if has_explicit_allocations:
        # Use explicit allocations
        for class_obj in classes:
            allocations = db.query(ClassTeacherAllocation).filter(
                ClassTeacherAllocation.class_id == class_obj.id
            ).all()
            
            for allocation in allocations:
                teacher_id = allocation.teacher_id
                subject_id = allocation.subject_id
                
                # Add to teacher_subjects if not already present
                if subject_id not in teacher_subjects[teacher_id]:
                    teacher_subjects[teacher_id].append(subject_id)
                
                # Add to subject_teachers if not already present
                if teacher_id not in subject_teachers[subject_id]:
                    subject_teachers[subject_id].append(teacher_id)
        
        # For teachers without explicit allocations, allow them to teach all subjects
        for teacher in all_teachers:
            if teacher.id not in teacher_subjects:
                teacher_subjects[teacher.id] = all_subject_ids.copy()
                # Update subject_teachers for this teacher
                for subject_id in all_subject_ids:
                    if teacher.id not in subject_teachers[subject_id]:
                        subject_teachers[subject_id].append(teacher.id)
        
        # For subjects without teachers, allow all teachers to teach them
        for subject in all_subjects:
            if subject.id not in subject_teachers:
                subject_teachers[subject.id] = [teacher.id for teacher in all_teachers]
                # Update teacher_subjects for these teachers
                for teacher in all_teachers:
                    if subject.id not in teacher_subjects[teacher.id]:
                        teacher_subjects[teacher.id].append(subject.id)
    
    else:
        # No explicit allocations - all teachers can teach all subjects
        for teacher in all_teachers:
            teacher_subjects[teacher.id] = all_subject_ids.copy()
        
        for subject in all_subjects:
            subject_teachers[subject.id] = [teacher.id for teacher in all_teachers]
    
    return dict(teacher_subjects), dict(subject_teachers)


def distribute_subjects_to_classes(
    classes: List[Class],
    all_subjects: List[Subject],
    subject_teachers: Dict[str, List[str]]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Distribute subjects to classes for the week.
    
    Returns: {class_id: {subject_id: [teacher_ids]}}
    
    Logic:
    - For each class, assign a set of subjects to be taught
    - Ensure variety of subjects across the week
    - Consider teacher availability for each subject
    """
    class_subject_assignments = defaultdict(lambda: defaultdict(list))
    
    if not all_subjects:
        return dict(class_subject_assignments)  # No subjects to distribute
    
    # Minimum and maximum subjects per class per week
    min_subjects_per_class = 5
    max_subjects_per_class = 8
    
    # Ensure we don't ask for more subjects than available
    available_subjects_count = len(all_subjects)
    actual_min_subjects = min(min_subjects_per_class, available_subjects_count)
    actual_max_subjects = min(max_subjects_per_class, available_subjects_count)
    
    # For each class
    for class_obj in classes:
        # Determine how many subjects this class should have
        if actual_min_subjects == actual_max_subjects:
            num_subjects = actual_min_subjects
        else:
            num_subjects = random.randint(actual_min_subjects, actual_max_subjects)
        
        # Select random subjects for this class
        if num_subjects > available_subjects_count:
            num_subjects = available_subjects_count
        
        if num_subjects <= 0:
            continue  # No subjects to assign
        
        selected_subjects = random.sample(all_subjects, num_subjects)
        
        # For each selected subject, assign available teachers
        for subject in selected_subjects:
            available_teachers = subject_teachers.get(subject.id, [])
            if available_teachers:
                # Select one or more teachers for this subject
                num_teachers = min(2, len(available_teachers))
                if num_teachers > 0:
                    selected_teachers = random.sample(available_teachers, num_teachers)
                    class_subject_assignments[class_obj.id][subject.id] = selected_teachers
    
    return dict(class_subject_assignments)

def generate_periods_for_class_v2(
    db: Session,
    class_obj: Class,
    class_schedule: ClassSchedule,
    class_subject_assignments: Dict[str, Dict[str, List[str]]],
    teacher_availability: Dict[str, Dict[str, List[int]]],
    days_of_week: List[str],
    period_timings: List[Tuple[time, time]]
) -> List[PeriodTimetable]:
    """
    Generate period timetable for a specific class using distributed subject assignments.
    
    Algorithm:
    1. Get assigned subjects and teachers for this class
    2. For each day, assign subjects to periods while ensuring:
       - Teacher is available at that time
       - No teacher has two classes at the same time
       - Subjects are distributed evenly across the week
    3. Track teacher usage to avoid conflicts
    """
    assigned_periods = []
    
    # Get subject assignments for this class
    subject_assignments = class_subject_assignments.get(class_obj.id, {})
    if not subject_assignments:
        return assigned_periods  # No subjects assigned to this class
    
    # Create a list of (subject_id, teacher_id) pairs for this class
    teacher_subject_pairs = []
    for subject_id, teacher_ids in subject_assignments.items():
        for teacher_id in teacher_ids:
            teacher_subject_pairs.append((teacher_id, subject_id))
    
    # Create a schedule for the week
    weekly_schedule = {}
    
    # For each day of the week
    for day in days_of_week:
        # Determine how many periods this class has on this day
        if class_schedule.schedule_type == ScheduleType.CUSTOM:
            # Get day-specific period count
            day_periods_map = {
                DayOfWeek.MONDAY.value: class_schedule.monday_periods,
                DayOfWeek.TUESDAY.value: class_schedule.tuesday_periods,
                DayOfWeek.WEDNESDAY.value: class_schedule.wednesday_periods,
                DayOfWeek.THURSDAY.value: class_schedule.thursday_periods,
                DayOfWeek.FRIDAY.value: class_schedule.friday_periods,
                DayOfWeek.SATURDAY.value: class_schedule.saturday_periods,
                DayOfWeek.SUNDAY.value: class_schedule.sunday_periods
            }
            periods_today = day_periods_map.get(day, 0)
        else:
            # For non-custom schedules, use periods_per_day
            periods_today = class_schedule.periods_per_day
        
        if periods_today <= 0:
            continue  # No periods on this day
        
        weekly_schedule[day] = {
            "total_periods": periods_today,
            "assigned_periods": []
        }
    
    # Distribute subjects across days
    days_with_periods = [day for day, info in weekly_schedule.items() if info["total_periods"] > 0]
    
    if not days_with_periods:
        return assigned_periods
    
    # Shuffle teacher-subject pairs to randomize assignment
    random.shuffle(teacher_subject_pairs)
    
    # Assign subjects to days
    current_pair_index = 0
    for day in days_with_periods:
        periods_today = weekly_schedule[day]["total_periods"]
        
        for period_idx in range(periods_today):
            if current_pair_index >= len(teacher_subject_pairs):
                current_pair_index = 0  # Reset if we run out of pairs
            
            teacher_id, subject_id = teacher_subject_pairs[current_pair_index]
            
            # Check if teacher is available at this period time
            if (teacher_id in teacher_availability and 
                day in teacher_availability[teacher_id] and 
                period_idx in teacher_availability[teacher_id][day]):
                
                weekly_schedule[day]["assigned_periods"].append({
                    "period_idx": period_idx,
                    "teacher_id": teacher_id,
                    "subject_id": subject_id
                })
                
                # Mark teacher as unavailable for this period
                if period_idx in teacher_availability[teacher_id][day]:
                    teacher_availability[teacher_id][day].remove(period_idx)
                
                current_pair_index += 1
            else:
                # Teacher not available, mark as break
                weekly_schedule[day]["assigned_periods"].append({
                    "period_idx": period_idx,
                    "is_break": True
                })
    
    # Create PeriodTimetable objects from the weekly schedule
    for day, day_info in weekly_schedule.items():
        for period_info in day_info["assigned_periods"]:
            period_idx = period_info["period_idx"]
            
            if period_info.get("is_break", False):
                # Create break period
                period = PeriodTimetable(
                    id=str(uuid.uuid4()),
                    class_schedule_id=class_schedule.id,
                    period_number=period_idx + 1,
                    day_of_week=DayOfWeek(day),
                    start_time=period_timings[period_idx][0],
                    end_time=period_timings[period_idx][1],
                    is_break=True,
                    break_name="Break",
                    is_active=True
                )
            else:
                # Create teaching period
                period = PeriodTimetable(
                    id=str(uuid.uuid4()),
                    class_schedule_id=class_schedule.id,
                    period_number=period_idx + 1,
                    day_of_week=DayOfWeek(day),
                    start_time=period_timings[period_idx][0],
                    end_time=period_timings[period_idx][1],
                    subject_id=period_info["subject_id"],
                    teacher_id=period_info["teacher_id"],
                    is_break=False,
                    is_active=True
                )
            
            assigned_periods.append(period)
    
    return assigned_periods


# =========================================================
# 🔹 AUTO GENERATE PERIODS API (UPDATED LOGIC)
# =========================================================

@router.post("/generate/auto/", response_model=dict, status_code=status.HTTP_201_CREATED)
def auto_generate_periods(
    school_id: str = Form(...),
    academic_year: str = Form(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Auto-generate period timetable for all classes in a school.
    
    NEW LOGIC: Teachers can teach any class without explicit allocations.
    
    This endpoint:
    1. Gets school configuration (start time, end time, period duration, etc.)
    2. Gets all classes with schedules
    3. Gets all teachers and subjects in the school
    4. Distributes subjects to classes for the week
    5. Generates period timetable ensuring no teacher conflicts
    6. Deletes existing periods for the school if they exist
    7. Creates new period timetable entries
    
    Algorithm Overview:
    - Calculate period timings based on school configuration
    - Get all teachers and subjects in the school
    - Distribute subjects to classes (5-8 subjects per class per week)
    - Create teacher availability matrix
    - For each class with schedule:
        - Assign teachers to periods based on subject distribution
        - Ensure no teacher has two classes at the same time
        - Distribute subjects evenly across the week
    """
    try:
        # =============================
        # 1️⃣ VALIDATIONS
        # =============================
        
        # Validate school exists and is active
        school = db.query(School).filter(
            School.id == school_id,
            School.is_active == True
        ).first()
        
        if not school:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="School not found or inactive"
            )

        # Get school configuration
        school_config = get_school_config(db, school_id)

        # =============================
        # 2️⃣ GET ALL CLASSES WITH SCHEDULES
        # =============================
        
        # Get all active classes in the school for the academic year
        classes = db.query(Class).filter(
            Class.school_id == school_id,
            Class.academic_year == academic_year,
            Class.is_active == True
        ).all()
        
        if not classes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active classes found for school {school.name} in academic year {academic_year}"
            )

        # Check which classes have schedules
        classes_with_schedules = []
        classes_without_schedules = []
        
        for class_obj in classes:
            # Check if class has active schedule
            schedule = db.query(ClassSchedule).filter(
                ClassSchedule.class_id == class_obj.id,
                ClassSchedule.is_active == True
            ).first()
            
            if not schedule:
                classes_without_schedules.append(class_obj.full_name)
                continue
            
            classes_with_schedules.append({
                "class": class_obj,
                "schedule": schedule
            })

        if not classes_with_schedules:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No classes with schedules found. Classes without schedules: {', '.join(classes_without_schedules)}"
            )
        # print("classes_with_schedules", classes_with_schedules)
        # print("classes_without_schedules", classes_without_schedules)
        # =============================
        # 3️⃣ GET ALL TEACHERS AND SUBJECTS
        # =============================
        
        # Get all teachers in the school
        teachers = db.query(User).filter(
            User.school_id == school_id,
            User.is_active == True,
            User.role.in_([UserRole.TEACHER, UserRole.HOD])
        ).all()
        
        if not teachers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No teachers found in the school"
            )

        # Get all subjects in the school
        all_subjects = get_all_subjects_for_school(db, school_id)
        # print("all_subjects",all_subjects)
        if not all_subjects:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subjects found in the school. Please create subjects first."
            )

        # =============================
        # 4️⃣ DISTRIBUTE SUBJECTS TO CLASSES
        # =============================
        
        # Get teacher-subject assignments
        teacher_subjects, subject_teachers = get_teacher_subject_assignments(
            db=db,
            school_id=school_id,
            classes=classes
        )
        # print("teacher_subjects", teacher_subjects)
        # print("subject_teachers", subject_teachers)
        # Distribute subjects to classes for the week
        class_subject_assignments = distribute_subjects_to_classes(
            classes=[info["class"] for info in classes_with_schedules],
            all_subjects=all_subjects,
            subject_teachers=subject_teachers
        )
        # print(class_subject_assignments, "class_subject_assignments")
        # =============================
        # 5️⃣ CALCULATE PERIOD TIMINGS
        # =============================
        
        # Days of week to generate periods for
        days_of_week = [day.value for day in DayOfWeek]
        
        # Calculate period timings based on school configuration
        # Use maximum periods per day from all class schedules
        max_periods_per_day = max(
            schedule["schedule"].periods_per_day 
            for schedule in classes_with_schedules
        )
        
        period_timings = calculate_period_timings(
            school_start_time=school_config.school_start_time,
            school_end_time=school_config.school_end_time,
            period_duration=school_config.period_duration,
            break_duration=school_config.break_duration,
            total_periods=max_periods_per_day
        )
        print("period_timings", period_timings)
        # =============================
        # 6️⃣ CREATE TEACHER AVAILABILITY MATRIX
        # =============================
        
        teacher_availability = get_teacher_availability_matrix(
            db=db,
            school_id=school_id,
            days_of_week=days_of_week,
            period_timings=period_timings
        )
        print("teacher_availability", teacher_availability)
        # =============================
        # 7️⃣ DELETE EXISTING PERIODS FOR THE SCHOOL
        # =============================
        
        # Find all period timetables for this school
        periods_to_delete = db.query(PeriodTimetable).join(
            ClassSchedule, PeriodTimetable.class_schedule_id == ClassSchedule.id
        ).join(
            Class, ClassSchedule.class_id == Class.id
        ).filter(
            Class.school_id == school_id,
            Class.academic_year == academic_year
        ).all()
        
        delete_count = len(periods_to_delete)
        print("periods_to_delete", periods_to_delete)
        
        if delete_count > 0:
            # Delete all existing periods
            for period in periods_to_delete:
                db.delete(period)
            db.commit()

        # =============================
        # 8️⃣ GENERATE NEW PERIODS FOR EACH CLASS
        # =============================
        
        all_generated_periods = []
        failed_classes = []
        print("classes_with_schedules", classes_with_schedules)
        for class_info in classes_with_schedules:
            try:
                generated_periods = generate_periods_for_class_v2(
                    db=db,
                    class_obj=class_info["class"],
                    class_schedule=class_info["schedule"],
                    class_subject_assignments=class_subject_assignments,
                    teacher_availability=teacher_availability,
                    days_of_week=days_of_week,
                    period_timings=period_timings
                )
                
                all_generated_periods.extend(generated_periods)
                
            except Exception as e:
                failed_classes.append({
                    "class": class_info["class"].full_name,
                    "error": str(e)
                })
        
        print("all_generated_periods", all_generated_periods)
        # =============================
        # 9️⃣ SAVE GENERATED PERIODS TO DATABASE
        # =============================
        
        success_count = 0
        for period in all_generated_periods:
            try:
                print("Period Object:")
                print(f"  ID: {period.id}")
                print(f"  Class Schedule ID: {period.class_schedule_id}")
                print(f"  Period Number: {period.period_number}")
                print(f"  Day of Week: {period.day_of_week}")
                print(f"  Start Time: {period.start_time}")
                print(f"  End Time: {period.end_time}")
                print(f"  Subject ID: {period.subject_id}")
                print(f"  Teacher ID: {period.teacher_id}")
                print(f"  Is Break: {period.is_break}")
                print(f"  Break Name: {period.break_name}")
                print(f"  Is Active: {period.is_active}")
                print("-" * 50)
                db.add(period)
                success_count += 1
            except Exception as e:
                # Log error but continue with other periods
                print(f"Error saving period: {e}")
        
        db.commit()

        # =============================
        # 🔟 RESPONSE
        # =============================
        print(success_count)
        # Prepare summary of subject distribution
        subject_distribution_summary = {}
        for class_id, subject_assignments in class_subject_assignments.items():
            class_obj = next((c["class"] for c in classes_with_schedules if c["class"].id == class_id), None)
            if class_obj:
                subject_names = []
                for subject_id in subject_assignments.keys():
                    subject = next((s for s in all_subjects if s.id == subject_id), None)
                    if subject:
                        subject_names.append(subject.name)
                
                subject_distribution_summary[class_obj.full_name] = {
                    "subject_count": len(subject_assignments),
                    "subjects": subject_names
                }
        
        response_data = {
            "school_id": school_id,
            "school_name": school.name,
            "academic_year": academic_year,
            "period_generation_summary": {
                "total_classes": len(classes),
                "classes_with_schedules": len(classes_with_schedules),
                "classes_without_schedules": len(classes_without_schedules),
                "total_teachers": len(teachers),
                "total_subjects": len(all_subjects),
                "existing_periods_deleted": delete_count,
                "new_periods_generated": success_count,
                "failed_classes": failed_classes
            },
            "subject_distribution": subject_distribution_summary,
            "school_timings": {
                "start_time": school_config.school_start_time.strftime("%H:%M"),
                "end_time": school_config.school_end_time.strftime("%H:%M"),
                "period_duration": school_config.period_duration,
                "break_duration": school_config.break_duration
            },
            "period_timings": [
                {
                    "period": idx + 1,
                    "start_time": start.strftime("%H:%M"),
                    "end_time": end.strftime("%H:%M")
                }
                for idx, (start, end) in enumerate(period_timings)
            ]
        }

        return success_response(
            message="Period timetable generated successfully",
            data=response_data,
            status_code=status.HTTP_201_CREATED
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate period timetable: {str(e)}"
        )


# =========================================================
# 🔹 DELETE ALL PERIODS FOR SCHOOL API
# =========================================================

@router.delete("/delete-all/", response_model=dict)
def delete_all_periods_for_school(
    school_id: str = Form(...),
    academic_year: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Delete all period timetables for a specific school.
    
    This endpoint removes all period timetable entries for:
    - A specific school
    - Optionally filtered by academic year
    
    Use with caution as this operation cannot be undone.
    """
    try:
        # =============================
        # 1️⃣ VALIDATIONS
        # =============================
        
        # Validate school exists
        school = db.query(School).filter(School.id == school_id).first()
        if not school:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="School not found"
            )

        # =============================
        # 2️⃣ FIND PERIODS TO DELETE
        # =============================
        
        # Build query to find periods
        query = db.query(PeriodTimetable).join(
            ClassSchedule, PeriodTimetable.class_schedule_id == ClassSchedule.id
        ).join(
            Class, ClassSchedule.class_id == Class.id
        ).filter(
            Class.school_id == school_id
        )
        
        # Filter by academic year if provided
        if academic_year:
            query = query.filter(Class.academic_year == academic_year)
        
        periods_to_delete = query.all()
        delete_count = len(periods_to_delete)
        
        if delete_count == 0:
            return success_response(
                message="No periods found to delete",
                data={
                    "school_id": school_id,
                    "academic_year": academic_year,
                    "periods_deleted": 0
                }
            )

        # =============================
        # 3️⃣ DELETE PERIODS
        # =============================
        
        for period in periods_to_delete:
            db.delete(period)
        
        db.commit()

        # =============================
        # 4️⃣ RESPONSE
        # =============================
        
        return success_response(
            message=f"Successfully deleted {delete_count} periods",
            data={
                "school_id": school_id,
                "school_name": school.name,
                "academic_year": academic_year,
                "periods_deleted": delete_count,
                "deleted_at": datetime.now().isoformat()
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete periods: {str(e)}"
        )
        
        
        
        
        
        
        
        