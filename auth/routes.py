from flask import Blueprint, render_template, redirect, url_for, flash, session
from .forms import LoginForm

# Create a Blueprint for authentication
auth = Blueprint('auth', __name__)

# Dummy user data for demonstration purposes
USER_CREDENTIALS = {
    "admin": "password123",
    "user": "userpass"
}

# Login route
@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session['user'] = username  # Store username in session
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template('login.html', form=form)

# Home route (protected page)
@auth.route('/home')
def home():
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))
    return render_template('home.html')

# Logout route
@auth.route('/logout')
def logout():
    session.pop('user', None)  # Clear session data
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
