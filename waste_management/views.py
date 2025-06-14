from csv import Error
from django.shortcuts import HttpResponse, redirect, render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import streamlit as st
import os
import plotly.figure_factory as ff
import pymysql
import warnings
import plotly.express as px
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from django.views.decorators.csrf import csrf_protect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from django.utils import timezone
from datetime import datetime, timedelta
from django.http import JsonResponse
from datetime import date
# Suppress warnings
warnings.filterwarnings('ignore')

# Load the trained model for waste classification
model = load_model('C:/Users/maste/Downloads/waste_management_system/waste_classification_model.h5')
categories = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
print("Model shape" , model.input_shape)
# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'supersaiyan1000',
    'database': 'prebooking',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def mess_login(request):
    if request.method == "POST":
        mess_id = request.POST.get("mess_id")
        password = request.POST.get("password")
        
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            cursor.execute("select mess_worker_name , password from mess_workers where mess_worker_name = %s and password = %s",(mess_id,password))
            mess_l = cursor.fetchone()
            print("mess chef : " , mess_l)
            if mess_l is not None:
                return redirect("mess_interface")
            else:
                messages.error(request, "Mess Worker doesn't exist. Please try Again")
                return redirect("mess_login")
        
    return render(request, "mess_login.html")


def mess_interface(request):
    today = datetime.now().date()
    next_days = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]
    bookings = []
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Fetch all users
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()

            for date in next_days:
                day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")

                for user in users:
                    user_id = user["id"]

                    # Get prebooking if available
                    cursor.execute("""
                        SELECT * FROM prebookings
                        WHERE id=%s AND date=%s AND on_leave=0
                    """, (user_id, date))

                    prebookings = cursor.fetchall()

                    if prebookings:
                        for pb in prebookings:
                            bookings.append({
                                "name": user["name"],
                                "roll_no": user["roll_no"],
                                "meal_type": pb["meal_type"],
                                "items": pb["meals"],
                                "quantity": pb["quantity"],
                                "date": pb["date"],
                                "source": "Prebooking"
                            })
                    else:
                        # No prebooking; get default diet
                        cursor.execute("""
                            SELECT * FROM default_food
                            WHERE id=%s AND day=%s AND hostel_id=%s
                        """, (user_id, day_name, user["hostel"]))

                        defaults = cursor.fetchall()

                        for df in defaults:
                            bookings.append({
                                "name": user["name"],
                                "roll_no": user["roll_no"],
                                "meal_type": df["meal_type"],
                                "items": df["items"],
                                "quantity": df["quantity"],
                                "date": date,
                                "source": "Default Diet"
                            })

    finally:
        connection.close()

    return render(request, "mess_interface.html", {"bookings": bookings})
    

def signup(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        roll_no = request.POST.get('roll_no')
        year = request.POST.get('year')
        course = request.POST.get('course')
        email_id = request.POST.get('email_id')
        hostel = request.POST.get('hostel')
        room_no = request.POST.get('room_no')
        contact_no = request.POST.get('contact_no')
        password = request.POST.get('password')

        if not all([name, roll_no, year, course, email_id, hostel, room_no, contact_no, password]):
            messages.error(request, "All fields are required.")
            return render(request, 'signup.html')

        try:
            year = int(year)
            if year < 1 or year > 5:
                messages.error(request, "Year must be between 1 and 5.")
                return render(request, 'signup.html')
        except ValueError:
            messages.error(request, "Year must be a valid number.")
            return render(request, 'signup.html')

        try:
            connection = pymysql.connect(**db_config)
            with connection.cursor() as cursor:
                cursor.execute("SELECT roll_no FROM users WHERE email_id = %s", (email_id,))
                if cursor.fetchone():
                    messages.error(request, "Roll number or email already exists.")
                    return render(request, 'signup.html')

                cursor.execute(
                    """
                    INSERT INTO users (name, roll_no, year, course, email_id, hostel, room_no, contact_no, password)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (name, roll_no, year, course, email_id, hostel, room_no, contact_no, password)
                )
                connection.commit()
                messages.success(request, "Registration successful! Please log in.")
                return redirect('login')
        except pymysql.Error as e:
            messages.error(request, f"Error during registration: {str(e)}")
            return render(request, 'signup.html')
        finally:
            if 'connection' in locals():
                connection.close()

    return render(request, 'signup.html')

def login(request):
    if request.method == 'POST':
        roll_no = request.POST.get('roll_no')
        password = request.POST.get('password')

        if not (roll_no and password):
            messages.error(request, "Roll number and password are required.")
            return render(request, 'login.html')

        try:
            connection = pymysql.connect(**db_config)
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE roll_no = %s AND password = %s", (roll_no, password))
                user = cursor.fetchone()
                if user:
                    request.session['roll_no'] = roll_no
                    messages.success(request, "Login successful!")
                    return redirect('home')
                else:
                    messages.error(request, "Invalid roll number or password.")
                    return render(request, 'login.html')
        except pymysql.Error as e:
            messages.error(request, f"Error during login: {str(e)}")
            return render(request, 'login.html')
        finally:
            if 'connection' in locals():
                connection.close()

    return render(request, 'login.html')
def community(request):
    connection = pymysql.connect(**db_config)
    
    # Always try to get existing posts first
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, date, name, blog_post, title FROM community ORDER BY date DESC, id DESC")
            blog_posts = cursor.fetchall()
    except Exception as e:
        print(f"Error fetching posts: {e}")
        blog_posts = []
    
    if request.method == "POST":
        title = request.POST.get("title")
        story = request.POST.get("content")
        
        if title and story:  # Only proceed if both fields have content
            try:
                with connection.cursor() as cursor:
                    # Get user's name
                    cursor.execute("SELECT name FROM users WHERE roll_no = %s", (request.session.get("roll_no")))
                    name = cursor.fetchone()
                    
                    if name:  # Only proceed if user exists
                        # Check if this exact post already exists from this user
                        cursor.execute(
                            """SELECT id FROM community 
                               WHERE name = %s AND title = %s AND blog_post = %s 
                               ORDER BY date DESC LIMIT 1""",
                            (name["name"], title, story)
                        )
                        existing_post = cursor.fetchone()
                        
                        if not existing_post:  # Only insert if it doesn't exist
                            cursor.execute(
                                """INSERT INTO community (date, name, blog_post, title) 
                                   VALUES (%s, %s, %s, %s)""",
                                (date.today(), name["name"], story, title)
                            )
                            connection.commit()
                            messages.success(request, "Story submitted successfully")
                        else:
                            messages.info(request, "You've already submitted this story")
            except Exception as e:
                print(f"Error inserting post: {e}")
                messages.error(request, f"Something went wrong... [{e}]")
        
        # Redirect after POST to prevent duplicate submissions
        return redirect('community')  # Make sure you have a URL named 'community'
    
    connection.close()
    return render(request, 'community.html', {'blog_post': blog_posts})

def mess_complaint_interface(request):
    pass


def predict_waste_type(img_path, model=model, categories=categories, target_size=(256, 256)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = (img_array / 127.5) - 1  # Normalize

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = categories[predicted_class]
        class_probabilities = {categories[i]: float(prob) for i, prob in enumerate(prediction[0])}

        return predicted_label, class_probabilities
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

def waste_classification_view(request):
    print("request method is :", request.method)
    if request.method == 'POST' and 'waste_image' in request.FILES:
        try:
            waste_image = request.FILES['waste_image']
            if not waste_image.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                messages.error(request, "Please upload an image file (PNG, JPG, JPEG)")
                return render(request, 'upload_image.html')

            print("yeah it is working")
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(waste_image.name, waste_image)
            image_path = os.path.join(settings.MEDIA_ROOT, filename)

            if not os.path.exists(image_path):
                messages.error(request, "Failed to save uploaded image.")
                return render(request, 'upload_image.html')

            prediction, confidence = predict_waste_type(img_path=image_path)
            print("prediction is :" , prediction)
            context = {
                'prediction': prediction,
                'image_url': fs.url(filename),
                'confidence': confidence,
                'top_confidence': max(confidence.values(), default=0)
            }
            return render(request, 'classification_result.html', context)
        except Exception as e:
            messages.error(request, f"Error processing image: {str(e)}")
            return render(request, 'upload_image.html')

    return render(request, 'upload_image.html')

def custom_login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if user.is_superuser:
                return redirect('admin:index')
            return render(request, 'index.html')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def generater_page(request):
    return render(request, 'GPT.html')

def homepage(request):
    roll_no = request.session.get('roll_no', None)
    if not roll_no:
        messages.error(request, "Please log in to access the homepage.")
        return redirect('login')

    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE roll_no = %s", (roll_no,))
            user = cursor.fetchone()
            if user is None:
                messages.error(request, "User not found. Please log in again.")
                return redirect('login')
            return render(request, 'index.html', {'user': user['name']})
    except pymysql.Error as e:
        messages.error(request, f"Database error: {str(e)}")
        return redirect('login')
    finally:
        if 'connection' in locals():
            connection.close()

def Waste_FAQ(request):
    return render(request, 'FAQ.html')

def Waste_Schedule(request):
    if request.method == "POST":
        task_name = request.POST.get("task_name")
        task_date = request.POST.get("task_date")
        task_time = request.POST.get("task_time")
        task_details = request.POST.get("task_details")
        print("task_name is :",task_name , " ",type(task_name))
        print("task_date is : ",task_date)
        
        connection = pymysql.connect(**db_config)
        try:
            with connection.cursor() as cursor:
                cursor.execute("Insert into schedule_task (task_name , task_date , task_time , task_comments) values(%s ,%s ,%s , %s)",(task_name,task_date,task_time,task_details))
                messages.success(request, "Task added successfully!")
                connection.commit()
                print("Successfully inserted")
        except :
            # messages.error("Something went wrong")
            print("something went wrong")
    return render(request, 'schedule.html')

def sanitation_report(request):
    if request.method == "POST":
        comments = request.POST.get("comments")
        image = request.FILES.get("file")
        # task_name = request.GET.get("task_name") or request.session.get("task_name", None)
        
        # Basic validation
        if comments:
            if len(comments) > 1000:
                messages.error(request, "Comments must not exceed 1000 characters.")
            else:
                connection = None
                try:
                    connection = pymysql.connect(**db_config)
                    image_url = None
                    if image:
                        # Validate image size (2MB limit as per template)
                        # if image.size > 2 * 1024 * 1024:
                        #     messages.error(request, "Image size must not exceed 2MB.")
                        #     return render(request, 'report.html')
                        
                        # Save image to filesystem
                        fs = FileSystemStorage(location='media/complaints')
                        filename = fs.save(image.name, image)
                        # Generate absolute URL
                        image_url = request.build_absolute_uri(fs.url(os.path.join('complaints', filename)))
                        print("Image url is :",image_url)
                        print("date is ",date.today())
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute(
                                "INSERT INTO complaints (complaint_date, comments, image_path) VALUES (%s, %s, %s)",
                                (date.today(), comments, image_url)
                            )
                        connection.commit()
                    except Exception as e:
                        print(f"Error inserting complaint: {e}")
                        messages.error(request, f"Database insert error: {str(e)}")
                except Exception as e:
                    messages.error(request , "Something went wrong")
    return render(request, 'report.html')

def facts(request):
    return render(request, 'fact.html')

def Recycling_guide(request):
    return render(request, 'guide.html')

def waste_dashBoard(request):
    return render(request, "Waste_DashBoard.html")

def Waste_Quiz(request):
    return render(request, 'Quiz.html')

def chat_view(request):
    return render(request, 'GPT.html')

def custom_logout_view(request):
    request.session.flush()
    messages.success(request, "Logged out successfully.")
    return redirect('login')

def prebooking(request):
    num_penalties = 0
    menu_items = {'breakfast': [], 'lunch': [], 'dinner': []}
    prebooking_meals = None
    meal_times = {'breakfast': None, 'lunch': None, 'dinner': None}

    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            # Fetch user's hostel and ID
            cursor.execute("SELECT hostel, id FROM users WHERE roll_no = %s", (request.session.get('roll_no'),))
            user = cursor.fetchone()
            if not user:
                messages.error(request, "User not found. Please log in.")
                return redirect('login')
            hostel_name = user['hostel']
            user_id = user['id']

            # Fetch menu items and their times
            cursor.execute("SELECT meal_type, meal_name, meal_id, constraints, time FROM menu_items WHERE hostel_name = %s", (hostel_name,))
            rows = cursor.fetchall()
            for row in rows:
                if row['meal_type'] in menu_items:
                    menu_items[row['meal_type']].append(row)
                    if row['meal_type'] in meal_times and not meal_times[row['meal_type']]:
                        meal_times[row['meal_type']] = row['time']  # Store first time for each meal_type

            # Fetch penalty count
            cursor.execute("SELECT COUNT(*) as penalty_count FROM has_arrived WHERE has_arrived = 0 AND id = %s", (user_id,))
            num_penalties = cursor.fetchone()['penalty_count']
            print(f"Penalty count: {num_penalties}")  # Debug
            print("Sidebar date:", request.POST.get("sidebar-date"))  # Debug
            # Fetch prebooking meals for sidebar
            sidebar_date = request.POST.get("sidebar-date")
            if sidebar_date:
                if request.method == 'POST' and 'sidebar-date' in request.POST:
            # sidebar_date = request.POST.get('sidebar-date')
                    try:
                        connection = pymysql.connect(**db_config)
                        with connection.cursor() as cursor:
                            cursor.execute(
                                "SELECT meals, meal_type FROM prebookings WHERE date = %s AND id = %s",
                                (sidebar_date, user_id)
                            )
                            bookings = cursor.fetchall()  # Changed to fetchall() to get all records

                            if bookings:
                                # Group meals by meal type
                                grouped_bookings = {}
                                for booking in bookings:
                                    meal_type = booking['meal_type']
                                    if meal_type not in grouped_bookings:
                                        grouped_bookings[meal_type] = []
                                    grouped_bookings[meal_type].append(booking['meals'])

                                return JsonResponse({
                                    'status': 'success',
                                    'bookings': grouped_bookings
                                })
                            else:
                                return JsonResponse({
                                    'status': 'empty',
                                    'message': 'No bookings for this date'
                                })
                    except Exception as e:
                        return JsonResponse({
                            'status': 'error',
                            'message': str(e)
                        }, status=500)

                  # Debug
                # request.session['prebooking_meals'] = prebooking_meals
    except pymysql.Error as e:
        messages.error(request, f"Database error: {str(e)}")
        return render(request, 'prebooking.html', {
            'menu_items': menu_items,
            'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            'penalty': num_penalties,
            'prebooking_meals': prebooking_meals,
        })
    finally:
        if 'connection' in locals():
            connection.close()
    # prebooking_meals = request.session.get("prebooking_meals")
    print("Prebooking meals outside the function : ", prebooking_meals)
    if request.method == 'POST':
        action = request.POST.get('action')  # Changed from 'leave' to 'action' to match HTML
        print(f"Request method: {request.method}, Action: {action}, POST data: {request.POST}")  # Debug
#knsdklfklsld
        if action == 'leave':
            unavailable_date = request.POST.get('unavailable_date')
            if not unavailable_date:
                messages.error(request, "Please select a date for unavailability.")
                return render(request, 'prebooking.html', {
                    'menu_items': menu_items,
                    'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'penalty': num_penalties,
                    'prebooking_meals': prebooking_meals,
                })

            try:
                booking_date = datetime.strptime(unavailable_date, '%Y-%m-%d').date()
                today = timezone.now().date()
                min_date = today + timedelta(days=1)
                max_date = today + timedelta(days=3)

                if not (min_date <= booking_date <= max_date):
                    messages.error(request, "Please select a date between tomorrow and the next 3 days.")
                    return render(request, 'prebooking.html', {
                        'menu_items': menu_items,
                        'min_date': min_date.strftime('%Y-%m-%d'),
                        'max_date': max_date.strftime('%Y-%m-%d'),
                        'penalty': num_penalties,
                        'prebooking_meals': prebooking_meals,
                    })
            except ValueError:
                messages.error(request, "Invalid date format.")
                return render(request, 'prebooking.html', {
                    'menu_items': menu_items,
                    'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'penalty': num_penalties,
                    'prebooking_meals': prebooking_meals,
                })
            print("booking_date:", booking_date)  # Debug
            print("meal_times:", meal_times)  # Debug
            print("today:", today)  # Debug
            if booking_date == today:
                current_time = timezone.now().time()
                cutoff_time = meal_times.get(meal_type)
                print(f"Meal type: {meal_type}, Cutoff time: {cutoff_time}")  # Debug
                if cutoff_time:
                    try:
                        cutoff_hour, cutoff_minute, cutoff_second = map(int, cutoff_time.split(':'))
                        cutoff_time_obj = datetime.strptime(cutoff_time, '%H:%M:%S').time()
                        if current_time > cutoff_time_obj:
                            messages.error(
                                request,
                                f"Cannot book {meal_type} for today after {cutoff_time_obj.strftime('%I:%M %p')}."
                            )
                            return render(request, 'prebooking.html', {
                                'menu_items': menu_items,
                                'min_date': min_date.strftime('%Y-%m-%d'),
                                'max_date': max_date.strftime('%Y-%m-%d'),
                                'penalty': num_penalties,
                                'prebooking_meals': prebooking_meals,
                            })
                    except ValueError as e:
                        print(f"Time parsing error: {str(e)}")  # Debug
                        messages.error(request, f"Invalid time format for {meal_type} cutoff time.")
                        return render(request, 'prebooking.html', {
                            'menu_items': menu_items,
                            'min_date': min_date.strftime('%Y-%m-%d'),
                            'max_date': max_date.strftime('%Y-%m-%d'),
                            'penalty': num_penalties,
                            'prebooking_meals': prebooking_meals,
                        })
            try:
                connection = pymysql.connect(**db_config)
                with connection.cursor() as cursor:
                    cursor.execute("SELECT id FROM users WHERE roll_no = %s", (request.session.get('roll_no'),))
                    user = cursor.fetchone()
                    if not user:
                        messages.error(request, "User not found.")
                        return redirect('login')

                    # Insert leave record without incrementing penalty
                    cursor.execute(
                        """
                        INSERT INTO prebookings (id, fine, penalty, date, meal_type, meals, quantity, on_leave)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (user['id'], 0, num_penalties, unavailable_date, None, None, 0, 1)
                    )
                    connection.commit()
                    messages.success(request, f"You are marked as unavailable for {unavailable_date}.")
            except pymysql.Error as e:
                messages.error(request, f"Error saving leave: {str(e)}")
            finally:
                if 'connection' in locals():
                    connection.close()
            return redirect('prebooking')

        elif action == 'book':
            date_str = request.POST.get('date')
            meal_type = request.POST.get('meal_type')
            food_items = request.POST.getlist('food_items')
            quantities = request.POST.getlist('quantities')

            if not date_str or not meal_type or not food_items:
                messages.error(request, "Please fill all required fields.")
                return render(request, 'prebooking.html', {
                    'menu_items': menu_items,
                    'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'penalty': num_penalties,
                    'prebooking_meals': prebooking_meals,
                })

            try:
                booking_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                today = timezone.now().date()
                min_date = today + timedelta(days=1)
                max_date = today + timedelta(days=3)

                if not (min_date <= booking_date <= max_date):
                    messages.error(request, "Please select a date between tomorrow and the next 3 days.")
                    return render(request, 'prebooking.html', {
                        'menu_items': menu_items,
                        'min_date': min_date.strftime('%Y-%m-%d'),
                        'max_date': max_date.strftime('%Y-%m-%d'),
                        'penalty': num_penalties,
                        'prebooking_meals': prebooking_meals,
                    })
            except ValueError:
                messages.error(request, "Invalid date format.")
                return render(request, 'prebooking.html', {
                    'menu_items': menu_items,
                    'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'penalty': num_penalties,
                    'prebooking_meals': prebooking_meals,
                })

            if meal_type not in ['breakfast', 'lunch', 'dinner']:
                messages.error(request, "Invalid meal type.")
                return render(request, 'prebooking.html', {
                    'menu_items': menu_items,
                    'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'penalty': num_penalties,
                    'prebooking_meals': prebooking_meals,
                })
            print(f"Booking date: {booking_date}, Meal type: {meal_type}, Food items: {food_items}, Quantities: {quantities}")  # Debug
            # Time restriction for tomorrow's bookings
            if booking_date == min_date:  # Booking for tomorrow
                current_time = timezone.now().time()
                cutoff_time = meal_times.get(meal_type)
                if cutoff_time:
                    try:
                        cutoff_hour, cutoff_minute, cutoff_second = map(int, cutoff_time.split(':'))
                        cutoff_time_obj = datetime.strptime(cutoff_time, '%H:%M:%S').time()
                        if current_time > cutoff_time_obj:
                            messages.error(
                                request,
                                f"Cannot book {meal_type} for tomorrow after {cutoff_time_obj.strftime('%I:%M %p')}."
                            )
                            return render(request, 'prebooking.html', {
                                'menu_items': menu_items,
                                'min_date': min_date.strftime('%Y-%m-%d'),
                                'max_date': max_date.strftime('%Y-%m-%d'),
                                'penalty': num_penalties,
                                'prebooking_meals': prebooking_meals,
                            })
                    except ValueError:
                        messages.error(request, f"Invalid time format in menu_items for {meal_type}.")
                        return render(request, 'prebooking.html', {
                            'menu_items': menu_items,
                            'min_date': min_date.strftime('%Y-%m-%d'),
                            'max_date': max_date.strftime('%Y-%m-%d'),
                            'penalty': num_penalties,
                            'prebooking_meals': prebooking_meals,
                        })

            try:
                connection = pymysql.connect(**db_config)
                with connection.cursor() as cursor:
                    cursor.execute("SELECT id FROM users WHERE roll_no = %s", (request.session.get('roll_no'),))
                    user = cursor.fetchone()
                    if not user:
                        messages.error(request, "User not found.")
                        return redirect('login')

                    # Check penalty count
                    if num_penalties >= 3:
                        messages.error(request, "You have reached the maximum number of penalties. Prebooking is not allowed.")
                        return render(request, 'prebooking.html', {
                            'menu_items': menu_items,
                            'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                            'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
                            'penalty': num_penalties,
                            'prebooking_meals': prebooking_meals,
                        })

                    fine = 0
                    for item, quantity in zip(food_items, quantities):
                        cursor.execute(
                            """
                            INSERT INTO prebookings (id, fine, penalty, date, meal_type, meals, quantity, on_leave)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (user['id'], fine, num_penalties, date_str, meal_type, item, int(quantity), 0)
                        )
                    connection.commit()
                    messages.success(request, f"Your {meal_type} booking for {date_str} has been successfully placed!")
            except pymysql.Error as e:
                messages.error(request, f"Error saving booking: {str(e)}")
            finally:
                if 'connection' in locals():
                    connection.close()
            return redirect('prebooking')

    return render(request, 'prebooking.html', {
        'menu_items': menu_items,
        'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
        'penalty': num_penalties,
        'prebooking_meals': prebooking_meals,
    })

def mark_unavailable(request):
    if request.method == 'POST':
        unavailable_date = request.POST.get('unavailable_date')
        if not unavailable_date:
            messages.error(request, "Please select a date for unavailability.")
            return redirect('prebooking')

        try:
            booking_date = datetime.strptime(unavailable_date, '%Y-%m-%d').date()
            today = timezone.now().date()
            min_date = today + timedelta(days=1)
            max_date = today + timedelta(days=3)

            if not (min_date <= booking_date <= max_date):
                messages.error(request, "Please select a date between tomorrow and the next 3 days.")
                return redirect('prebooking')
        except ValueError:
            messages.error(request, "Invalid date format.")
            return redirect('prebooking')

        try:
            connection = pymysql.connect(**db_config)
            with connection.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE roll_no = %s", (request.session.get('roll_no'),))
                user = cursor.fetchone()
                if not user:
                    messages.error(request, "User not found.")
                    return redirect('login')

                cursor.execute("SELECT COUNT(*) as penalty_count FROM has_arrived WHERE has_arrived = 0 AND id = %s", (user['id'],))
                num_penalties = cursor.fetchone()['penalty_count']

                if num_penalties >= 3:
                    messages.error(request, "You have reached the maximum number of penalties. Prebooking is not allowed.")
                    return redirect('prebooking')

                cursor.execute(
                    """
                    INSERT INTO prebookings (id, fine, penalty, date, meal_type, meals, quantity, on_leave)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (user['id'], 0, num_penalties, unavailable_date, None, None, 0, 1)
                )
                connection.commit()
                messages.success(request, f"You are marked as unavailable for {unavailable_date}.")
        except pymysql.Error as e:
            messages.error(request, f"Error saving leave: {str(e)}")
        finally:
            if 'connection' in locals():
                connection.close()
        return redirect('prebooking')

    return redirect('prebooking')
import logging
logger = logging.getLogger(__name__)
def default_diet(request):
    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # Get user info
            cursor.execute("SELECT hostel, id FROM users WHERE roll_no = %s", 
                          (request.session.get('roll_no'),))
            user = cursor.fetchone()
            if not user:
                messages.error(request, "User not found. Please log in.")
                return redirect('login')
            
            hostel_name = user['hostel']
            user_id = user['id']
            logger.debug(f"User ID: {user_id}, Hostel: {hostel_name}")

            # Get menu items for dropdowns
            cursor.execute("SELECT meal_type, meal_name FROM menu_items WHERE hostel_name = %s", 
                          (hostel_name,))
            menu_items = {
                'breakfast': [],
                'lunch': [],
                'dinner': []
            }
            for row in cursor.fetchall():
                if row['meal_type'].lower() in menu_items:
                    menu_items[row['meal_type'].lower()].append(row)

            # Get current default diet
            cursor.execute("""
                SELECT day, meal_type, items 
                FROM default_food
                WHERE id = %s
                ORDER BY day, meal_type
            """, (user_id,))
            
            default_diet = {
                hostel_name.lower(): {
                    "monday": ["", "", ""], 
                    "tuesday": ["", "", ""], 
                    "wednesday": ["", "", ""], 
                    "thursday": ["", "", ""], 
                    "friday": ["", "", ""], 
                    "saturday": ["", "", ""], 
                    "sunday": ["", "", ""]
                }
            }
            
            rows = cursor.fetchall()
            if not rows:
                logger.warning(f"No default diet entries found for user_id: {user_id}")
            
            for row in rows:
                day = row['day'].lower()
                meal_type = row['meal_type'].lower()
                items = row['items'] or ""
                logger.debug(f"Day: {day}, Meal Type: {meal_type}, Items: {items}")
                if day in default_diet[hostel_name.lower()]:
                    meal_index = {'breakfast': 0, 'lunch': 1, 'dinner': 2}.get(meal_type)
                    if meal_index is not None:
                        default_diet[hostel_name.lower()][day][meal_index] = items
                    else:
                        logger.warning(f"Invalid meal_type: {meal_type} for day: {day}")
                else:
                    logger.warning(f"Invalid day: {day}")

            logger.debug(f"Default Diet: {default_diet}")

        return render(request, 'default_diet.html', {
            'default_diet': default_diet,
            'hostel': hostel_name,
            'menu_items': menu_items
        })
        
    except pymysql.Error as e:
        messages.error(request, f"Database error: {str(e)}")
        logger.error(f"Database error: {str(e)}")
        return redirect('default_diet')
    finally:
        if 'connection' in locals():
            connection.close()

def update_default_diet(request):
    if request.method == 'POST':
        try:
            connection = pymysql.connect(**db_config)
            with connection.cursor() as cursor:
                cursor.execute("SELECT hostel, id FROM users WHERE roll_no = %s", 
                          (request.session.get('roll_no'),))
                user = cursor.fetchone()
                if not user:
                    messages.error(request, "User not found. Please log in.")
                    return redirect('login')
                
                hostel_name = user['hostel']
                user_id = user['id']
                print("user id is ", user_id)
                # user_id = request.session.get('user_id')  # Adjust based on your auth
                for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                    breakfast = ','.join(request.POST.getlist(f'{day}_breakfast'))
                    lunch = ','.join(request.POST.getlist(f'{day}_lunch'))
                    dinner = ','.join(request.POST.getlist(f'{day}_dinner'))
                    
                    # Update or insert into default_food
                    for meal_type, items in [('breakfast', breakfast), ('lunch', lunch), ('dinner', dinner)]:
                        if items:
                            cursor.execute("""
                                INSERT INTO default_food (id, day, meal_type, items,hostel_id)
                                VALUES (%s, %s, %s, %s, %s)
                                ON DUPLICATE KEY UPDATE items = %s
                            """, (user_id, day, meal_type, items, hostel_name, items))
            connection.commit()
            messages.success(request, "Preferences updated successfully!")
        except pymysql.Error as e:
            messages.error(request, f"Database error: {str(e)}")
        finally:
            connection.close()
        return redirect('default_diet')
    return redirect('default_diet')