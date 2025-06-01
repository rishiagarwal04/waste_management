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

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the trained model for waste classification
model = load_model('C:/Users/maste/Downloads/waste_management_system/waste_management/waste_model_3.h5')
categories = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'supersaiyan1000',
    'database': 'prebooking',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

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
    return render(request, 'community.html')

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
    if request.method == 'POST' and 'waste_image' in request.FILES:
        try:
            waste_image = request.FILES['waste_image']
            if not waste_image.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                messages.error(request, "Please upload an image file (PNG, JPG, JPEG)")
                return render(request, 'upload_image.html')

            fs = FileSystemStorage()
            filename = fs.save(waste_image.name, waste_image)
            image_path = os.path.join(settings.MEDIA_ROOT, filename)

            if not os.path.exists(image_path):
                messages.error(request, "Failed to save uploaded image.")
                return render(request, 'upload_image.html')

            prediction, confidence = predict_waste_type(img_path=image_path)
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
    return render(request, 'schedule.html')

def sanitation_report(request):
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

            # Fetch prebooking meals for sidebar
            sidebar_date = request.POST.get("sidebar-date")
            if sidebar_date:
                cursor.execute(
                    "SELECT meals FROM prebookings WHERE date = %s AND id = %s",
                    (sidebar_date, user_id)
                )
                prebooking_meals = cursor.fetchone()

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

    if request.method == 'POST':
        action = request.POST.get('action')  # Changed from 'leave' to 'action' to match HTML
        print(f"Request method: {request.method}, Action: {action}, POST data: {request.POST}")  # Debug

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

def default_diet(request):
    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            cursor.execute("SELECT hostel FROM users WHERE roll_no = %s", (request.session.get('roll_no'),))
            user = cursor.fetchone()
            if not user:
                messages.error(request, "User not found.")
                return redirect('login')
            hostel_name = user['hostel']

            default_diet = {
                hostel_name.lower(): {
                    "monday": [], "tuesday": [], "wednesday": [], "thursday": [], "friday": [], "saturday": [], "sunday": []
                }
            }
            cursor.execute("SELECT day, meal_type, meal_name FROM default_diet WHERE hostel_name = %s", (hostel_name,))
            rows = cursor.fetchall()
            for row in rows:
                day = row['day'].lower()
                if day in default_diet[hostel_name.lower()]:
                    default_diet[hostel_name.lower()][day].append(f"{row['meal_type']}: {row['meal_name']}")

        return render(request, 'default_diet.html', {'default_diet': default_diet, 'hostel': hostel_name})
    except pymysql.Error as e:
        messages.error(request, f"Database error: {str(e)}")
        return redirect('home')
    finally:
        if 'connection' in locals():
            connection.close()