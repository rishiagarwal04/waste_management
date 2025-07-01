## Canteen Waste Management System

**The Canteen Waste Management System is a web-based application designed to reduce food waste in institutional canteens, such as those in hostels or universities, with a focus on sustainability and efficient resource management. Inspired by efforts at IIT Guwahati’s Kapili Hostel, this system helps track food waste, manage meal pre-booking, and promote strategies like the Just-in-Time (JIT) model and composting to minimize environmental and financial impacts.**


## Introduction

**The Canteen Waste Management System aims to address the significant issue of food waste in canteens, where, for example, Kapili Hostel at IIT Guwahati generates approximately 64.47 kg of daily food waste. By automating meal pre-booking, tracking waste metrics, and integrating sustainable practices like composting and JIT preparation, the system enhances operational efficiency, reduces waste, and fosters a culture of sustainability among students and staff.**

Features





**Meal Pre-booking: Allows students to pre-book meals, reducing overproduction via the JIT model.**



**Waste Tracking: Monitors food waste (e.g., plate waste, kitchen waste) with real-time data and visualizations.**



**Data Visualization: Displays waste metrics (e.g., daily waste in kg, meal type) using dashboards built with Streamlit or similar tools.**



**Sustainability Features: Supports composting, biogas production, and donation of edible leftovers to NGOs.**



**Reporting: Generates CSV reports for waste data, including day, meal type, and absence percentages.**



**User Management: Role-based access for students, mess managers, and admins to manage bookings and waste data.**





## Technologies Used


Frontend: HTML, CSS, JavaScript DOM , 



Backend: Python, Django, MySQL(pymysql) 



Data Visualization: Streamlit (for dashboards)



Others:  Git, Render/Vercel (for deployment)

Installation

To set up the Canteen Waste Management System locally, follow these steps:

Prerequisites





Python 3.8+




MySQL(pymysql) 



Git

Steps





Clone the Repository:

git clone https://github.com/rishiagarwal04/canteen-waste-management.git
cd canteen-waste-management



Set Up :

cd waste_management
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt



Configure Environment Variables: Create a .env file in the server directory:

DATABASE_URL=your_database_connection_string
SECRET_KEY=your_django_secret_key
PORT=8000



Set Up Database:





For MySQL: Create a database and update the DATABASE_URL.



Run migrations:

python manage.py migrate



Run the Application:

**python manage.py runserver**






## Usage



Pre-booking Meals: Students log in to pre-book meals, specifying meal type and quantity, reducing overproduction.



Waste Tracking: Mess managers input waste data (e.g., 59.57 kg breakfast plate waste) via forms, which are stored and visualized.



Data Analysis: Use the Streamlit dashboard to filter by meal type (breakfast, lunch, dinner) or export CSV reports.



Sustainability Actions: Implement composting or biogas production based on system recommendations.





Fork the repository.



Create a feature branch (git checkout -b feature/your-feature).



Commit changes (git commit -m "Add your feature").



Push to the branch (git push origin feature/your-feature).



Open a pull request.

Please review open issues on the GitHub issues page for tasks.

License

This project is licensed under the MIT License - see the LICENSE.md file for details.

Contact

For inquiries or collaboration, contact:





Email:rishiagarwal094a@gmail.com



GitHub: https://github.com/rishiagarwal04
