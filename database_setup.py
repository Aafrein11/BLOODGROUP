import sqlite3

# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('auth/user_data.db')
cursor = conn.cursor()

# Create the users table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')

# Insert sample user data (username: admin, password: admin123)
cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", ('admin', 'admin123'))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database setup completed successfully.")
