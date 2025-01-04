import sqlite3
import streamlit as st

# def get_database_connection():
#     """Create a database connection and return connection and cursor objects."""
#     conn = sqlite3.connect('user_data.db', check_same_thread=False)
#     c = conn.cursor()
#     return conn, c

# def initialize_database():
#     """Initialize SQLite database and create users table if it doesn't exist."""
#     try:
#         conn, c = get_database_connection()
        
#         # Create users table if it doesn't exist
#         def get_database_connection():
#     """Create a database connection and return connection and cursor objects."""
#     conn = sqlite3.connect('user_data.db', check_same_thread=False)
#     c = conn.cursor()
#     return conn, c

# import sqlite3
# import streamlit as st

# def get_database_connection():
#     """Create a database connection and return connection and cursor objects."""
#     conn = sqlite3.connect('user_data.db', check_same_thread=False)
#     c = conn.cursor()
#     return conn, c

# import sqlite3
# import streamlit as st

def get_database_connection():
    """Create a database connection and return connection and cursor objects."""
    conn = sqlite3.connect('user_data.db', check_same_thread=False)
    c = conn.cursor()
    return conn, c

def initialize_database():
    """Initialize SQLite database and create necessary tables if they don't exist."""
    try:
        conn, c = get_database_connection()
        
        # Create users table if it doesn't exist
        c.execute('''
            SELECT count(name) FROM sqlite_master 
            WHERE type='table' AND name='users'
        ''')
        
        if c.fetchone()[0] == 0:
            c.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    name TEXT NOT NULL CHECK(length(name) > 0),
                    email TEXT NOT NULL UNIQUE CHECK(length(email) > 0 AND email LIKE '%@%'),
                    age INTEGER NOT NULL CHECK(age >= 0 AND age <= 120),
                    sex TEXT NOT NULL CHECK(sex IN ('Male', 'Female', 'Other')),
                    password TEXT NOT NULL CHECK(length(password) >= 6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
                )
            ''')
            
            c.execute('''
                CREATE TRIGGER IF NOT EXISTS update_user_timestamp 
                AFTER UPDATE ON users
                BEGIN
                    UPDATE users SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            ''')

        # Create freshness_records table if it doesn't exist
        c.execute('''
            SELECT count(name) FROM sqlite_master 
            WHERE type='table' AND name='freshness_records'
        ''')
        
        if c.fetchone()[0] == 0:
            c.execute('''
                CREATE TABLE freshness_records (
                    sl_no INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    produce TEXT NOT NULL CHECK(length(produce) > 0),
                    freshness TEXT NOT NULL CHECK(length(freshness) > 0),
                    expected_life_span_days INTEGER 
                )
            ''')

        # Create ocr_records table if it doesn't exist
        c.execute('''
            SELECT count(name) FROM sqlite_master 
            WHERE type='table' AND name='ocr_records'
        ''')
        
        if c.fetchone()[0] == 0:
            c.execute('''
                CREATE TABLE ocr_records (
                    sl_no INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    product_name TEXT,
                    brand_name TEXT,
                    type_of_product TEXT,
                    batch_number TEXT,
                    year_of_manufacturing TEXT,
                    expiry_date TEXT,
                    other_relevant_details TEXT,
                    utilization_time TEXT,
                    count INTEGER DEFAULT 1,
                    expired TEXT,
                    expected_life_span_days TEXT
                )
            ''')
        
        # Create item_counting_records table if it doesn't exist
        c.execute('''
            SELECT count(name) FROM sqlite_master 
            WHERE type='table' AND name='item_counting_records'
        ''')
        
        if c.fetchone()[0] == 0:
            c.execute('''
                CREATE TABLE item_counting_records (
                    sl_no INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    fruits TEXT,
                    vegetables TEXT,
                    packed_goods TEXT,
                    beverages TEXT,
                    bakery_essentials TEXT,
                    others TEXT,
                    count INTEGER DEFAULT 1
                )
            ''')

        conn.commit()
        st.success("Database initialized successfully.")
    
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")
    finally:
        conn.close()


def insert_freshness_record(produce, freshness, expected_life_span_days):
    """Insert a new freshness record into the freshness_records table."""
    try:
        conn, c = get_database_connection()
        c.execute('''
            INSERT INTO freshness_records (produce, freshness, expected_life_span_days)
            VALUES (?, ?, ?)
        ''', (produce, freshness, expected_life_span_days))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error while inserting freshness record: {e}")
    finally:
        conn.close()

def create_account(name, email, age, sex, password):
    """Create a new user account in the SQLite database with validation."""
    try:
        # Input validation
        if not name or len(name.strip()) == 0:
            st.error("Name cannot be empty")
            return None
            
        if not email or '@' not in email:
            st.error("Invalid email format")
            return None
            
        if not isinstance(age, int) or age < 0 or age > 120:
            st.error("Age must be between 0 and 120")
            return None
            
        if sex not in ['Male', 'Female', 'Other']:
            st.error("Invalid sex selection")
            return None
            
        if not password or len(password) < 6:
            st.error("Password must be at least 6 characters long")
            return None

        conn, c = get_database_connection()
        
        # Check if email already exists
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        if c.fetchone() is not None:
            st.error("Email already exists!")
            return None
            
        # Insert new user
        c.execute('''
            INSERT INTO users (name, email, age, sex, password)
            VALUES (?, ?, ?, ?, ?)
        ''', (name.strip(), email.lower().strip(), age, sex, password))
        
        conn.commit()
        
        # Get the user info to return
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user_data = c.fetchone()
        
        if user_data:
            user_info = {
                "name": user_data[1],
                "email": user_data[2],
                "age": user_data[3],
                "sex": user_data[4],
                "password": user_data[5]
            }
            return user_info
        return None
        
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()
def insert_item_counting_record(items_data, total_count):
    """
    Insert a new item counting record into the item_counting_records table.
    
    Parameters:
    - items_data (dict): A dictionary containing item counts with keys:
        - 'Fruits'
        - 'Vegetables'
        - 'Packed Goods'
        - 'Beverages'
        - 'Bakery Essentials'
        - 'Others'
    - total_count (int): The total number of items detected.
    """
    try:
        conn, c = get_database_connection()
        
        # Extract items per category and join them into comma-separated strings
        fruits = ', '.join(items_data.get('Fruits', [])) if 'Fruits' in items_data and items_data['Fruits'] else None
        vegetables = ', '.join(items_data.get('Vegetables', [])) if 'Vegetables' in items_data and items_data['Vegetables'] else None
        packed_goods = ', '.join(items_data.get('Packed Goods', [])) if 'Packed Goods' in items_data and items_data['Packed Goods'] else None
        beverages = ', '.join(items_data.get('Beverages', [])) if 'Beverages' in items_data and items_data['Beverages'] else None
        bakery_essentials = ', '.join(items_data.get('Bakery Essentials', [])) if 'Bakery Essentials' in items_data and items_data['Bakery Essentials'] else None
        others = ', '.join(items_data.get('Others', [])) if 'Others' in items_data and items_data['Others'] else None
        
        # Insert the record with the count
        c.execute('''
            INSERT INTO item_counting_records (
                fruits,
                vegetables,
                packed_goods,
                beverages,
                bakery_essentials,
                others,
                count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            fruits,
            vegetables,
            packed_goods,
            beverages,
            bakery_essentials,
            others,
            total_count
        ))
        
        conn.commit()
        st.success("Item counting information saved to the database.")
    
    except sqlite3.Error as e:
        st.error(f"Database error while inserting item counting record: {e}")
    finally:
        conn.close()


def insert_ocr_record(product_info):
    """
    Insert a new OCR record into the ocr_records table.
    
    Parameters:
    - product_info (dict): A dictionary containing product information with keys:
        - 'Product Name'
        - 'Brand Name'
        - 'Type of Product'
        - 'Batch Number'
        - 'Year of Manufacturing'
        - 'Expiry Date'
        - 'Other relevant Details'
        - 'Utilization time'
        - 'Expired' (optional)
        - 'Expected life span (Days)' (optional)
    """
    try:
        conn, c = get_database_connection()
        
        # Extract information with default values as None
        product_name = product_info.get('Product Name')
        brand_name = product_info.get('Brand Name')
        type_of_product = product_info.get('Type of Product')
        batch_number = product_info.get('Batch Number')
        year_of_manufacturing = product_info.get('Year of Manufacturing')
        expiry_date = product_info.get('Expiry Date')
        other_relevant_details = product_info.get('Other relevant Details')
        utilization_time = product_info.get('Utilization time')
        expired = product_info.get('Expired')  # Optional
        expected_life_span_days = product_info.get('Expected life span (Days)')  # Optional
        
        # Insert the record
        c.execute('''
            INSERT INTO ocr_records (
                product_name,
                brand_name,
                type_of_product,
                batch_number,
                year_of_manufacturing,
                expiry_date,
                other_relevant_details,
                utilization_time,
                expired,
                expected_life_span_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            product_name,
            brand_name,
            type_of_product,
            batch_number,
            year_of_manufacturing,
            expiry_date,
            other_relevant_details,
            utilization_time,
            expired,
            expected_life_span_days
        ))
        
        conn.commit()
        st.success("Product information saved to the database.")
    
    except sqlite3.Error as e:
        st.error(f"Database error while inserting OCR record: {e}")
    finally:
        conn.close()

def check_login(username, password):
    """Check login credentials against SQLite database with validation."""
    try:
        if not username or not password:
            st.error("Username and password are required")
            return None
            
        conn, c = get_database_connection()
        
        # Check credentials
        c.execute('SELECT * FROM users WHERE LOWER(email) = LOWER(?) AND password = ?', 
                 (username.strip(), password))
        user_data = c.fetchone()
        
        if user_data:
            user_info = {
                "name": user_data[1],
                "email": user_data[2],
                "age": user_data[3],
                "sex": user_data[4],
                "password": user_data[5]
            }
            return user_info
        return None
        
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()
