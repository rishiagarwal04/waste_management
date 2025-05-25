from django.shortcuts import HttpResponse,redirect,render
from django.core.files.storage import FileSystemStorage
import pandas as pd 
import streamlit as st 
import pandas as pd 
import os
import plotly.figure_factory as ff
import pymysql
import warnings
import plotly.express as px
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer , AutoModelForCausalLM
# import bcrypt
# from .forms import UserRegisterForm
from django.views.decorators.csrf import csrf_protect
from django.contrib import messages
from django.contrib.auth import authenticate,login
warnings.filterwarnings('ignore')
# from .models import CustomerForm
# def customer_registration(request):
#     if request.method == 'POST':
#         form = CustomerForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('homepage')  # Redirect to a success page or another view
#     else:
#         form = CustomerForm()
from django.conf import settings
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('C:/Users/maste/Downloads/waste_management_system/waste_management/models/waste_model_3.h5')

# Define categories
categories = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']


def signup(request):
    # Database connection configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'supersaiyan1000',
        'database': 'prebooking',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }

    if request.method == 'POST':
        # Extract form data
        name = request.POST.get('name')
        roll_no = request.POST.get('roll_no')
        year = request.POST.get('year')
        course = request.POST.get('course')
        email_id = request.POST.get('email_id')
        hostel = request.POST.get('hostel')
        room_no = request.POST.get('room_no')
        contact_no = request.POST.get('contact_no')
        password = request.POST.get('password')

        # Validate input
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

        # Hash password
        # hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Save user to database
        try:
            connection = pymysql.connect(**db_config)
            with connection.cursor() as cursor:
                # Check if roll_no or email_id already exists
                cursor.execute("SELECT roll_no FROM users WHERE email_id = %s", (email_id))
                if cursor.fetchone():
                    messages.error(request, "Roll number or email already exists.")
                    return render(request, 'signup.html')

                # Insert new user
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
        # finally:
        #     if 'connection' in locals():
        #         connection.close()

    # For GET request, render the signup form
    return render(request, 'signup.html')
db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'supersaiyan1000',
        'database': 'prebooking',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }
def login(request):
    # Database connection configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'supersaiyan1000',
        'database': 'prebooking',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }

    if request.method == 'POST':
        request.session['roll_no'] = request.POST.get('roll_no')
        roll_no = request.session.get('roll_no', None)
        password = request.POST.get('password')

        # Validate input
        if not (roll_no and password):
            messages.error(request, "Roll number and password are required.")
            return render(request, 'login.html')

        # Authenticate user
        try:
            connection = pymysql.connect(**db_config)
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE roll_no = %s and password = %s", (roll_no,password))
                user = cursor.fetchone()
                if user:
                    # Store user ID in session
                    # request.session['id'] = user['id']
                    messages.success(request, "Login successful!")
                    return redirect('home')
                else:
                    messages.error(request, "Invalid roll number or password.")
                    return render(request, 'login.html')
        except pymysql.Error as e:
            messages.error(request, f"Error during login: {str(e)}")
            return render(request, 'login.html')
        # finally:
        #     if 'connection' in locals():
        #         connection.close()

    # For GET request, render the login form
    return render(request, 'login.html')
    
def community(request):
    return render(request,'community.html')

def predict_waste_type(img_path, model=model, categories=categories, target_size=(256, 256)):
    """
    Predict the waste category for the given image.
    
    Parameters:
        img_path (str): Absolute path to the input image.
        model (keras.Model): Trained CNN model for waste classification.
        categories (list): List of category labels.
        target_size (tuple): Target size for resizing the image.
    
    Returns:
        tuple: (predicted_label, class_probabilities)
    """
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)# Normalize

        new_img_array = np.expand_dims(new_img_array, axis=0)
        # new_img_array = (img_array / 127.5) -1# Normalize
        print("Pixel value range - before normalization : ",img_array.min(), " " , img_array.max())
        new_pred = model.predict(new_img_array)
        
        print("New Image Prediction:", new_pred)
        new_pred_class = np.argmax(new_pred, axis=1)[0]
        print("New Image Predicted Class:", categories[new_pred_class])
        
        # Predict using the model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Extract results
        predicted_label = categories[predicted_class]
        class_probabilities = {
            categories[i]: float(prob)  # Convert numpy float to Python float
            for i, prob in enumerate(prediction[0])
        }
        
        return predicted_label, class_probabilities
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

def waste_classification_view(request):
    if request.method == 'POST' and 'waste_image' in request.FILES:
        try:
            waste_image = request.FILES['waste_image']
            
            # Validate file type
            if not waste_image.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                messages.error(request, "Please upload an image file (PNG, JPG, JPEG)")
                return render(request, 'upload_image.html')
            
            # Save the file
            fs = FileSystemStorage()
            filename = fs.save(waste_image.name, waste_image)
            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            
            # Verify the file was saved
            if not os.path.exists(image_path):
                messages.error(request, "Failed to save uploaded image.")
                return render(request, 'upload_image.html')
            
            # Process the image
            prediction, confidence = predict_waste_type(img_path=image_path)
            
            # Prepare context for template
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
    
    return render(request, 'upload_image.html')
def custom_login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            # Custom redirect logic
            if user.is_superuser:
                return redirect('admin:index')  # Example redirect for admin users
            return render(request,'index.html') # Redirect to a specific page for regular users
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def generater_page(request):
    return render(request, 'GPT.html')
def homepage(request):
    roll_no = request.session.get('roll_no', None)
    query  = "select * from users where roll_no = %s"
    connection = pymysql.connect(**db_config)
    with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE roll_no = %s", (roll_no,))
            user = cursor.fetchone()
            if user is None:
                messages.error(request, "User not found. Please log in again.")
                return redirect('login')
            else:
                user = user["name"] # 
            return render(request , 'index.html', {'user': user})
            day = []
            cursor.execute()
def Waste_FAQ(request): 
    return render(request , 'FAQ.html')
def Waste_Schedule(request):
    return render(request,'schedule.html')
def sanitation_report(request):
    return render(request,'report.html')
def facts(request):
    return render(request , 'fact.html')
def Recycling_guide(request):
    return render(request , 'guide.html')
def waste_dashBoard(request):
    return render(request , "Waste_DashBoard.html")
def Waste_Quiz(request):
    return render(request,'Quiz.html')
def chat_view(request):
    return render(request , 'GPT.html')


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import timedelta, datetime
from django.contrib import messages
from .models import Prebooking, MenuItem


def custom_logout_view(request):
    pass 

# @login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils import timezone
from datetime import datetime, timedelta
import pymysql

def prebooking(request):
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'supersaiyan1000',
        'database': 'prebooking',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }

    # Fetch menu items grouped by meal type
    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            menu_items = {'breakfast': [], 'lunch': [], 'dinner': []}
            cursor.execute("SELECT meal_name, meal_type FROM menu_items")
            rows = cursor.fetchall()
            for row in rows:
                if row['meal_type'] in menu_items:
                    menu_items[row['meal_type']].append(row)
    except pymysql.Error as e:
        messages.error(request, f"Database error: {str(e)}")
        return render(request, 'prebooking.html', {
            'menu_items': {'breakfast': [], 'lunch': [], 'dinner': []},
            'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
        })
    finally:
        if 'connection' in locals():
            connection.close()

    num_penalties = 0  # default

    # POST Request
    if request.method == 'POST':
        date_str = request.POST.get('date')
        meal_type = request.POST.get('meal_type')
        food_items_ids = request.POST.getlist('food_items')

        # Validate date
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
                })
        except ValueError:
            messages.error(request, "Invalid date format.")
            return render(request, 'prebooking.html', {
                'menu_items': menu_items,
                'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            })

        if meal_type not in ['breakfast', 'lunch', 'dinner']:
            messages.error(request, "Invalid meal type.")
            return render(request, 'prebooking.html', {
                'menu_items': menu_items,
                'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            })

        if not food_items_ids:
            messages.error(request, "Please select at least one food item.")
            return render(request, 'prebooking.html', {
                'menu_items': menu_items,
                'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
            })

        # Save the prebooking
        try:
            connection = pymysql.connect(**db_config)
            with connection.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE roll_no = %s", (request.session.get('roll_no'),))
                user = cursor.fetchone()

                if not user:
                    messages.error(request, "User not found.")
                    return redirect('login')

                # Check penalty count
                cursor.execute("SELECT COUNT(*) as penalty_count FROM has_arrived WHERE has_arrived = 0 AND id = %s", (user['id'],))
                has_arrived_count = cursor.fetchone()
                num_penalties = has_arrived_count['penalty_count']

                if num_penalties >= 3:
                    messages.error(request, "You have already missed 3 meals. You cannot prebook any more meals.")
                    return render(request, 'prebooking.html', {
                        'menu_items': menu_items,
                        'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
                        'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
                        'penalty': num_penalties
                    })

                fine = 0
                penalty = num_penalties

                cursor.execute(
                    "INSERT INTO prebookings (id, fine, penalty, date, meal_type) VALUES (%s, %s, %s, %s, %s)",
                    (user["id"], fine, penalty, booking_date, meal_type)
                )

                connection.commit()
                messages.success(request, f"Your {meal_type} booking for {booking_date} has been successfully placed!")
                return redirect('prebooking')

        except pymysql.Error as e:
            messages.error(request, f"Error saving your booking: {str(e)}")
        finally:
            if 'connection' in locals():
                connection.close()

    return render(request, 'prebooking.html', {
        'menu_items': menu_items,
        'min_date': (timezone.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'max_date': (timezone.now().date() + timedelta(days=3)).strftime('%Y-%m-%d'),
        'penalty': num_penalties
    })


def default_diet(request):
    """View to display the default diet"""
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'supersaiyan1000',
        'database': 'prebooking',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }
    
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()
    
    
    day = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    hostel = ["Lohit","kapili ","dhansiri","subansiri","kameng"]
    default_diet = {
        "lohit":{"monday": [], "tuesday": [], "wednesday": [], "thursday": [], "friday": [], "saturday": [], "sunday": []},
        "kapili": {"monday": [], "tuesday": [], "wednesday": [], "thursday": [], "friday": [], "saturday": [], "sunday": []},
        "dhansiri": {"monday": [], "tuesday": [], "wednesday": [], "thursday": [], "friday": [], "saturday": [], "sunday": []},
        "subansiri": {"monday": [], "tuesday": [], "wednesday": [], "thursday": [], "friday": [], "saturday": [], "sunday": []},
        "kameng": {"monday": [], "tuesday": [], "wednesday": [], "thursday": [], "friday": [], "saturday": [], "sunday": []}
    }
    

# def mark_attendance(request):
#     """Admin view to mark attendance and apply penalties"""
#     if not request.user.is_staff:
#         return redirect('login')

#     db_config = {
#         'host': 'localhost',
#         'user': 'root',
#         'password': 'supersaiyan1000',
#         'database': 'prebooking',
#         'charset': 'utf8mb4',
#         'cursorclass': pymysql.cursors.DictCursor
#     }

#     if request.method == 'POST':
#         try:
#             connection = pymysql.connect(**db_config)
#             with connection.cursor() as cursor:
#                 # Get today's date
#                 today = timezone.now().date().strftime('%Y-%m-%d')
                
#                 # Get all prebookings for today without attendance marked
#                 cursor.execute("""
#                     SELECT p.id
#                     FROM prebookings p
#                     LEFT JOIN has_arrived h ON p.id = h.prebooking_id
#                     WHERE p.date = %s AND h.prebooking_id IS NULL
#                 """, (today,))
#                 unmarked = cursor.fetchall()

#                 for booking in unmarked:
#                     # Default to not arrived (penalty)
#                     cursor.execute("""
#                         INSERT INTO has_arrived (prebooking_id, has_arrived)
#                         VALUES (%s, 0)
#                     """, (booking['id'],))
                    
#                     # Increase penalty count
#                     cursor.execute("""
#                         UPDATE users 
#                         SET penalty = penalty + 1 
#                         WHERE id = %s AND penalty < 3
#                     """, (booking['id'],))
                    
#                     # Add fine if reached 3 penalties
#                     cursor.execute("""
#                         UPDATE users 
#                         SET fine = 1000 
#                         WHERE id = %s AND penalty >= 3 AND fine = 0
#                     """, (booking['id'],))
                
#                 connection.commit()
#                 messages.success(request, "Attendance marked and penalties applied.")
#         except pymysql.Error as e:
#             messages.error(request, f"Error: {str(e)}")
#         finally:
#             if 'connection' in locals():
#                 connection.close()

#     return render(request, 'admin/mark_attendance.html')